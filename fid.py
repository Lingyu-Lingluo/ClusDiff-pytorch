from cleanfid import fid

fdir1 = './data/food-101'
fdir2 = './generated_images'

score = fid.compute_fid(fdir1, fdir2)
clip_score = fid.compute_fid(fdir1, fdir2, mode='clean', model_name='clip_vit_b_32')
print(f'FID score:{score} | CLIP-FID score:{clip_score}')

