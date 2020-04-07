import csv
import pickle
import os

ROOT_DIR = '/media/data/movie/dataset/tf_records/'

csv_fi = open(os.path.join(ROOT_DIR, 'results.csv'), mode='w')
writer = csv.writer(
    csv_fi,
    delimiter=',',
    quotechar='|',
    quoting=csv.QUOTE_MINIMAL)
writer.writerow(['experiment', 'model_name', 'mean_L2_repeat_first',\
            'mean_L2_repeat_last', 'mean_L2_weighted',\
            'mean_L2_inter_frames', 'mean_psnr_repeat_first',\
            'mean_psnr_repeat_last', 'mean_psnr_weighted',\
            'mean_psnr_inter_frames'])

window_exp = os.listdir(ROOT_DIR)
window_exp = [
    i
    for i in window_exp
    if not i.endswith('csv')]

for window in window_exp:
    models = os.listdir(os.path.join(
        ROOT_DIR, window))

    models = [
        i
        for i in models
        if not i.endswith('pkl') and not i.endswith('records')]

    for model in models:
        files = os.listdir(os.path.join(
            ROOT_DIR, window, model))

        pkl_file = [
            i
            for i in files
            if i.endswith('pkl')]

        if len(pkl_file):
            pkl_file = pkl_file[0]

            pkl_path = os.path.join(
                ROOT_DIR, window, model, pkl_file)

            with open(pkl_path, 'rb') as handle:
                eval_pkl = pickle.load(handle)

            writer.writerow([window, model,
                eval_pkl['mean_repeat_first'],\
                eval_pkl['mean_repeat_last'],\
                eval_pkl['mean_weighted_frames'],\
                eval_pkl['mean_inter_frames'],\
                eval_pkl['mean_psnr_repeat_first'],\
                eval_pkl['mean_psnr_repeat_last'],\
                eval_pkl['mean_psnr_weighted_frames'],\
                eval_pkl['mean_psnr_inter_frames']])

            print('Finished storing {} model\'s info'.format(model))


csv_fi.close()
print('Process finished......')

