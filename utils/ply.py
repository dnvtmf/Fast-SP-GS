import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ['load_ply', 'save_ply', 'parse_ply']

_ply_dtype_map = {
    'char': ('b', int, np.int8),
    'int8': ('b', int, np.int8),
    'uchar': ('B', int, np.uint8),
    'uint8': ('B', int, np.uint8),
    'short': ('h', int, np.int16),
    'int16': ('h', int, np.int16),
    'ushort': ('H', int, np.uint16),
    'uint16': ('H', int, np.uint16),
    'int': ('i', int, np.int32),
    'int32': ('i', int, np.int32),
    'uint': ('I', int, np.uint32),
    'uint32': ('I', int, np.uint32),
    'float': ('f', float, np.float32),
    'float32': ('f', float, np.float32),
    'double': ('d', float, np.float64),
    'float64': ('d', float, np.float64),
}  # type: Dict[str, Tuple[str, type(int), np.dtype]]


def _ply_get_format(dtypes, fmt: str):
    if fmt == 'binary_little_endian':
        fmt = '<'
    elif fmt == 'binary_big_endian':
        fmt = '>'
    else:
        assert fmt == 'ascii'
        return None
    for x in dtypes:
        fmt += _ply_dtype_map[x][0]
    return fmt, struct.calcsize(fmt)


def parse_ply(filename):
    """ parse a ply file
    reference: http://gamma.cs.unc.edu/POWERPLANT/papers/ply.pdf
    """
    error_msg = f"can not load ply file {filename}"
    with open(filename, 'rb') as f:
        ##### load header
        line_no = 0
        ply_format = None
        data = []
        while True:
            items = f.readline().decode('ascii').split()
            # print(items)
            if line_no == 0:
                assert len(items) == 1 and items[0] == 'ply', error_msg
            elif len(items) == 0 or items[0] == 'comment':
                continue
            elif items[0] == 'format':
                assert len(items) == 3 and items[2] == '1.0', error_msg
                ply_format = items[1]
                assert ply_format in ['ascii', 'binary_little_endian', 'binary_big_endian']
            elif items[0] == 'end_header':
                assert len(items) == 1, error_msg
                break
            elif items[0] == 'element':
                assert len(items) == 3, error_msg
                data.append({
                    'name': items[1],
                    'num': int(items[2]),
                    'dtypes': [],
                    'names': [],
                    'all_scalar': True,
                    'data': [],
                })
            elif items[0] == 'property':
                if items[1] == 'list':
                    assert len(items) == 5, error_msg
                    num_dtype = items[2]
                    data_type = items[3]
                    assert len(data) > 0
                    data[-1]['names'].append(items[-1])
                    data[-1]['dtypes'].append((num_dtype, data_type))
                    data[-1]['all_scalar'] = False
                else:
                    assert len(items) == 3, error_msg
                    data_type = items[1]
                    assert len(data) > 0
                    data[-1]['names'].append(items[-1])
                    data[-1]['dtypes'].append(data_type)

                assert data_type in [
                    'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double', 'int8', 'uint8', 'int16',
                    'uint16', 'int32', 'uint32', 'float32', 'float64'
                ], error_msg
            else:
                raise NotImplementedError('Undealed header:', items)
            line_no += 1
        assert ply_format is not None, error_msg
        ########## load context
        for elem in data:
            num = len(elem['names'])
            elem['data'] = [[] for _ in range(num)]
            if ply_format == 'ascii':
                for line_no in range(elem['num']):
                    values = f.readline().split()
                    row = []
                    i = 0
                    for dtype in elem['dtypes']:
                        if isinstance(dtype, str):
                            row.append(_ply_dtype_map[dtype][1](values[i].decode('ascii')))
                            i += 1
                        else:
                            num = int(values[i].decode('ascii'))
                            i += 1
                            row.append((_ply_dtype_map[dtype[1]][1](values[i + j].decode('ascii')) for j in range(num)))
                            i += num
                    assert len(row) == num
                    for i in range(num):
                        elem['data'][i].append(row[i])
                continue
            if elem['all_scalar']:
                x_formant, length = _ply_get_format(elem['dtypes'], ply_format)
                x_data = f.read(length * elem['num'])
                assert len(x_data) == length * elem['num']
                elem['data'] = list(zip(*struct.iter_unpack(x_formant, x_data)))
            else:
                fmts = []
                dtypes = []
                for dtype in elem['dtypes']:
                    if isinstance(dtype, str):
                        dtypes.append(dtype)
                    else:
                        dtypes.append(dtype[0])
                        fmts.append(_ply_get_format(dtypes, ply_format))
                        fmts.append(_ply_get_format([dtype[1]], ply_format))
                        dtypes.clear()
                if len(dtypes) > 0:
                    fmts.append(_ply_get_format(dtypes, ply_format))
                for line in range(elem['num']):
                    row = []
                    for i, (fmt, length) in enumerate(fmts):
                        if i % 2 == 0:
                            row.extend(struct.unpack_from(fmt, f.read(length)))
                        else:
                            cnt = row.pop(-1)
                            row.append(struct.unpack_from(f"{fmt[0]}{cnt}{fmt[1]}", f.read(length * cnt)))
                    for i in range(num):
                        elem['data'][i].append(row[i])
            # convert to numpy
            for i in range(num):
                if isinstance(elem['dtypes'][i], str):
                    dtype = elem['dtypes'][i]
                    elem['data'][i] = np.array(elem['data'][i], dtype=_ply_dtype_map[dtype][2])
                else:
                    len_data0 = len(elem['data'][i][0])
                    if all(len(x) == len_data0 for x in elem['data'][i]):
                        dtype = elem['dtypes'][i][1]
                        elem['data'][i] = np.array(elem['data'][i], dtype=_ply_dtype_map[dtype][2])
    return data


def load_ply(filename: Path):
    # parse to mesh format
    mesh_data = parse_ply(filename)

    def _to_array(element: dict, names: list, *extra, axis=-1):
        for name in names:
            assert name in element['names']
        indices = [element['names'].index(name) for name in names]
        indices.extend(element['names'].index(name) for name in extra if name in element['names'])
        dtype = element['dtypes'][indices[0]]
        for index in indices:
            assert element['dtypes'][index] == dtype
        return np.stack([element['data'][i] for i in indices], axis=axis)

    data = {}
    for elem in mesh_data:
        if elem['name'] == 'vertex':
            data['v_pos'] = _to_array(elem, ['x', 'y', 'z'])
            if 'nx' in elem['names']:
                data['v_nrm'] = _to_array(elem, ['nx', 'ny', 'nz'])
            if 'red' in elem['names']:
                data['v_clr'] = _to_array(elem, ['red', 'green', 'blue'], 'alpha')
            for name in elem['names']:
                if name not in ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha']:
                    print(f'ply undealed atte {name} for vertex')
        elif elem['name'] == 'face':
            f_pos = []
            faces = elem['data'][elem['names'].index('vertex_indices')]
            # print(type(faces), faces.shape)
            if isinstance(faces, np.ndarray):
                if faces.shape[1] == 3:
                    data['f_pos'] = faces
                else:
                    for i in range(1, faces.shape[1] - 1):
                        f_pos.append(np.stack([faces[:, 0], faces[:, i], faces[:, i + 1]], axis=-1))
                    data['f_pos'] = np.stack(f_pos, axis=1).reshape(-1, 3)
            else:
                for ploy in faces:
                    for i in range(1, len(ploy) - 1):
                        f_pos.append((ploy[0], ploy[i], ploy[i + 1]))
                dtype = elem['dtypes'][elem['names'].index('vertex_indices')][1]
                data['f_pos'] = np.array(f_pos, dtype=_ply_dtype_map[dtype][2])
            for name in elem['names']:
                if name != 'vertex_indices':
                    print('ply undealed element', name)
        else:
            print('ply undealed element', elem['name'])
    data = {k: torch.from_numpy(v) for k, v in data.items()}
    return data


def save_ply(file, data):
    import trimesh
    if not isinstance(data, trimesh.Trimesh):
        assert len(data) == 2
        vertices, triangles = data
        if isinstance(vertices, Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(triangles, Tensor):
            triangles = triangles.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    else:
        mesh = data

    mesh.export(str(file))
    # fout = open(name, 'w')
    # fout.write("ply\n")
    # fout.write("format ascii 1.0\n")
    # fout.write("element vertex " + str(len(vertices)) + "\n")
    # fout.write("property float x\n")
    # fout.write("property float y\n")
    # fout.write("property float z\n")
    # fout.write("element face " + str(len(triangles)) + "\n")
    # fout.write("property list uchar int vertex_index\n")
    # fout.write("end_header\n")
    # for ii in range(len(vertices)):
    #     fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    # for ii in range(len(triangles)):
    #     fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    # fout.close()
    return
