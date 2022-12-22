#%%
import torch
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
import numpy as np
from fastapi import FastAPI
import uvicorn
from fastapi.responses import StreamingResponse
from io import BytesIO


api = FastAPI()


# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

print('creating SDF model...')
sdf_model = model_from_config(MODEL_CONFIGS["sdf"], device)
sdf_model.eval()

print('loading SDF model...')
sdf_model.load_state_dict(load_checkpoint("sdf", device))


def mesh_to_binmodel(mesh):
    positions = mesh.verts
    # print(f"positions: {positions.shape}")

    indices = mesh.faces.reshape(mesh.faces.shape[0] * mesh.faces.shape[1])
    print(f"indices: {indices.shape}")

    normals = mesh.normals
    # print(f"normals: {normals.shape}")

    colors = np.stack([
        mesh.vertex_channels["R"], 
        mesh.vertex_channels["G"], 
        mesh.vertex_channels["B"],
        # np.full(mesh.vertex_channels["R"].shape, 1.0)
    ], axis=1)
    # print(f"colors: {colors.shape}")

    all = np.concatenate([
        positions, 
        normals,
        colors
    ], axis=1)
    print(f"all: {all.shape}")

    data = bytearray()
    indices_len = np.ascontiguousarray(len(indices), dtype=">u2").tobytes()
    data.extend(indices_len)
    all_len = np.ascontiguousarray(len(all), dtype=">u2").tobytes()
    data.extend(all_len)
    indices_bytes = np.ascontiguousarray(indices, dtype=">u2").tobytes()
    data.extend(indices_bytes)
    all_bytes = np.ascontiguousarray(all, dtype=">f4").tobytes()
    data.extend(all_bytes)

    return data

    # # write data to file
    # with open("mesh.bin", "wb") as f:
    #     f.write(data)


def prompt_to_point_cloud(prompt):
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]
    return pc


def point_cloud_to_mesh(pc):
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=32, # increase to 128 for resolution used in evals
        progress=True,
    )
    return mesh


@api.get("/gen")
def gen(prompt: str):
    pc = prompt_to_point_cloud(prompt)
    # pc = PointCloud.load('example_data/pc_corgi.npz')
    mesh = point_cloud_to_mesh(pc)
    data = mesh_to_binmodel(mesh)
    # fig = plot_point_cloud(pc, grid_size=2)

    # bytearray to bytesio
    bytes = BytesIO()
    bytes.write(data)
    bytes.seek(0)

    return StreamingResponse(bytes)


uvicorn.run(api, host="0.0.0.0", port=7870)