import argparse
import os
from time import gmtime, strftime

import numpy as np
import torch

import cls_feature_class
import parameters
import seldnet_model
def get_accdoa_labels(accdoa_in, nb_classes):
    x = accdoa_in[:, :, :nb_classes]
    y = accdoa_in[:, :, nb_classes:2 * nb_classes]
    z = accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    x0 = accdoa_in[:, :, :1 * nb_classes]
    y0 = accdoa_in[:, :, 1 * nb_classes:2 * nb_classes]
    z0 = accdoa_in[:, :, 2 * nb_classes:3 * nb_classes]
    dist0 = accdoa_in[:, :, 3 * nb_classes:4 * nb_classes]
    dist0[dist0 < 0.] = 0.0
    sed0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2) > 0.5
    doa0 = accdoa_in[:, :, :3 * nb_classes]

    x1 = accdoa_in[:, :, 4 * nb_classes:5 * nb_classes]
    y1 = accdoa_in[:, :, 5 * nb_classes:6 * nb_classes]
    z1 = accdoa_in[:, :, 6 * nb_classes:7 * nb_classes]
    dist1 = accdoa_in[:, :, 7 * nb_classes:8 * nb_classes]
    dist1[dist1 < 0.] = 0.0
    sed1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) > 0.5
    doa1 = accdoa_in[:, :, 4 * nb_classes:7 * nb_classes]

    x2 = accdoa_in[:, :, 8 * nb_classes:9 * nb_classes]
    y2 = accdoa_in[:, :, 9 * nb_classes:10 * nb_classes]
    z2 = accdoa_in[:, :, 10 * nb_classes:11 * nb_classes]
    dist2 = accdoa_in[:, :, 11 * nb_classes:]
    dist2[dist2 < 0.] = 0.0
    sed2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) > 0.5
    doa2 = accdoa_in[:, :, 8 * nb_classes:11 * nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def _distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    # 출력 단위: 도(degree)
    eps = 1e-10
    num = x1 * x2 + y1 * y2 + z1 * z2
    den = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) + eps
    val = np.clip(num / den, -1.0, 1.0)
    return np.arccos(val) * 180.0 / np.pi


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        dist = _distance_between_cartesian_coordinates(
            doa_pred0[class_cnt],
            doa_pred0[class_cnt + 1 * nb_classes],
            doa_pred0[class_cnt + 2 * nb_classes],
            doa_pred1[class_cnt],
            doa_pred1[class_cnt + 1 * nb_classes],
            doa_pred1[class_cnt + 2 * nb_classes],
        )
        return 1 if dist < thresh_unify else 0
    return 0


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def _get_loc_feat(params):
    if params['dataset'] == 'mic':
        if params['use_salsalite']:
            return '{}_salsa'.format(params['dataset'])
        return '{}_gcc'.format(params['dataset'])
    return params['dataset']


def _get_loc_output(params):
    return 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'


def _get_dev_splits(params):
    if '2020' in params['dataset_dir']:
        return [1]
    if '2021' in params['dataset_dir']:
        return [6]
    if '2022' in params['dataset_dir']:
        return [[4]]
    if '2023' in params['dataset_dir']:
        return [[4]]
    if '2024' in params['dataset_dir']:
        return [[4]]
    raise ValueError('Unknown dataset splits for dataset_dir: {}'.format(params['dataset_dir']))


def _resolve_split(params, split_idx):
    if params['mode'] == 'eval':
        return None
    test_splits = _get_dev_splits(params)
    if split_idx < 0 or split_idx >= len(test_splits):
        raise ValueError('split_idx {} out of range [0, {})'.format(split_idx, len(test_splits)))
    return test_splits[split_idx]


def _build_model_path(params, task_id, job_id, split_idx, override_path=''):
    if override_path:
        return override_path
    unique_name = '{}_{}_{}_split{}_{}_{}'.format(
        task_id,
        job_id,
        params['mode'],
        split_idx,
        _get_loc_output(params),
        _get_loc_feat(params),
    )
    return '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))


def _collect_file_list(feat_dir, mode, split):
    files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    if mode == 'eval':
        return sorted(files)

    split_set = set(np.array(split).reshape(-1).tolist())
    out = []
    for fname in files:
        if len(fname) > 4:
            try:
                fold_id = int(fname[4])
            except ValueError:
                continue
            if fold_id in split_set:
                out.append(fname)
    return sorted(out)


def _load_feature_sequences(feat_path, feature_seq_len, nb_mel_bins):
    feat = np.load(feat_path)
    if feat.ndim != 2:
        raise ValueError('Expected 2D feature array, got {}'.format(feat.shape))

    if feat.shape[1] % nb_mel_bins != 0:
        raise ValueError('Feature dim {} not divisible by nb_mel_bins {}'.format(feat.shape[1], nb_mel_bins))

    nb_ch = feat.shape[1] // nb_mel_bins
    n_frames = feat.shape[0]
    nb_seq = int(np.ceil(n_frames / float(feature_seq_len)))
    pad_frames = nb_seq * feature_seq_len - n_frames

    if pad_frames > 0:
        feat = np.vstack((feat, np.ones((pad_frames, feat.shape[1]), dtype=feat.dtype) * 1e-6))

    feat = feat.reshape(feat.shape[0], nb_ch, nb_mel_bins)
    feat = feat.reshape(nb_seq, feature_seq_len, nb_ch, nb_mel_bins)
    feat = np.transpose(feat, (0, 2, 1, 3)).astype(np.float32)
    return feat


def _run_model(model, feat_tensor, device):
    with torch.no_grad():
        x = torch.tensor(feat_tensor, dtype=torch.float32, device=device)
        out = model(x)
    return out.detach().cpu().numpy()


def _compute_uncertainty(output_np, nb_classes, multi_accdoa, low, high):
    if multi_accdoa:
        norms = []
        for track_idx in range(3):
            base = track_idx * 4 * nb_classes
            x = output_np[:, :, base:base + nb_classes]
            y = output_np[:, :, base + nb_classes:base + 2 * nb_classes]
            z = output_np[:, :, base + 2 * nb_classes:base + 3 * nb_classes]
            norms.append(np.sqrt(x * x + y * y + z * z))
        norm_stack = np.concatenate(norms, axis=-1)
    else:
        x = output_np[:, :, :nb_classes]
        y = output_np[:, :, nb_classes:2 * nb_classes]
        z = output_np[:, :, 2 * nb_classes:3 * nb_classes]
        norm_stack = np.sqrt(x * x + y * y + z * z)

    ambiguous = (norm_stack >= low) & (norm_stack <= high)
    return float(np.mean(ambiguous))


def _output_to_dict(output_np, params):
    output_dict = {}
    if params['multi_accdoa'] is True:
        sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output_np, params['unique_classes'])
        sed_pred0 = reshape_3Dto2D(sed_pred0)
        doa_pred0 = reshape_3Dto2D(doa_pred0)
        dist_pred0 = reshape_3Dto2D(dist_pred0)
        sed_pred1 = reshape_3Dto2D(sed_pred1)
        doa_pred1 = reshape_3Dto2D(doa_pred1)
        dist_pred1 = reshape_3Dto2D(dist_pred1)
        sed_pred2 = reshape_3Dto2D(sed_pred2)
        doa_pred2 = reshape_3Dto2D(doa_pred2)
        dist_pred2 = reshape_3Dto2D(dist_pred2)

        for frame_cnt in range(sed_pred0.shape[0]):
            for class_cnt in range(sed_pred0.shape[1]):
                flag_0sim1 = determine_similar_location(
                    sed_pred0[frame_cnt][class_cnt],
                    sed_pred1[frame_cnt][class_cnt],
                    doa_pred0[frame_cnt],
                    doa_pred1[frame_cnt],
                    class_cnt,
                    params['thresh_unify'],
                    params['unique_classes'],
                )
                flag_1sim2 = determine_similar_location(
                    sed_pred1[frame_cnt][class_cnt],
                    sed_pred2[frame_cnt][class_cnt],
                    doa_pred1[frame_cnt],
                    doa_pred2[frame_cnt],
                    class_cnt,
                    params['thresh_unify'],
                    params['unique_classes'],
                )
                flag_2sim0 = determine_similar_location(
                    sed_pred2[frame_cnt][class_cnt],
                    sed_pred0[frame_cnt][class_cnt],
                    doa_pred2[frame_cnt],
                    doa_pred0[frame_cnt],
                    class_cnt,
                    params['thresh_unify'],
                    params['unique_classes'],
                )

                if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                    if sed_pred0[frame_cnt][class_cnt] > 0.5:
                        output_dict.setdefault(frame_cnt, []).append([
                            class_cnt,
                            doa_pred0[frame_cnt][class_cnt],
                            doa_pred0[frame_cnt][class_cnt + params['unique_classes']],
                            doa_pred0[frame_cnt][class_cnt + 2 * params['unique_classes']],
                            dist_pred0[frame_cnt][class_cnt],
                        ])
                    if sed_pred1[frame_cnt][class_cnt] > 0.5:
                        output_dict.setdefault(frame_cnt, []).append([
                            class_cnt,
                            doa_pred1[frame_cnt][class_cnt],
                            doa_pred1[frame_cnt][class_cnt + params['unique_classes']],
                            doa_pred1[frame_cnt][class_cnt + 2 * params['unique_classes']],
                            dist_pred1[frame_cnt][class_cnt],
                        ])
                    if sed_pred2[frame_cnt][class_cnt] > 0.5:
                        output_dict.setdefault(frame_cnt, []).append([
                            class_cnt,
                            doa_pred2[frame_cnt][class_cnt],
                            doa_pred2[frame_cnt][class_cnt + params['unique_classes']],
                            doa_pred2[frame_cnt][class_cnt + 2 * params['unique_classes']],
                            dist_pred2[frame_cnt][class_cnt],
                        ])

                elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                    output_dict.setdefault(frame_cnt, [])
                    if flag_0sim1:
                        if sed_pred2[frame_cnt][class_cnt] > 0.5:
                            output_dict[frame_cnt].append([
                                class_cnt,
                                doa_pred2[frame_cnt][class_cnt],
                                doa_pred2[frame_cnt][class_cnt + params['unique_classes']],
                                doa_pred2[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                dist_pred2[frame_cnt][class_cnt],
                            ])
                        doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                        output_dict[frame_cnt].append([
                            class_cnt,
                            doa_pred_fc[class_cnt],
                            doa_pred_fc[class_cnt + params['unique_classes']],
                            doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                            dist_pred_fc[class_cnt],
                        ])

                    elif flag_1sim2:
                        if sed_pred0[frame_cnt][class_cnt] > 0.5:
                            output_dict[frame_cnt].append([
                                class_cnt,
                                doa_pred0[frame_cnt][class_cnt],
                                doa_pred0[frame_cnt][class_cnt + params['unique_classes']],
                                doa_pred0[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                dist_pred0[frame_cnt][class_cnt],
                            ])
                        doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                        output_dict[frame_cnt].append([
                            class_cnt,
                            doa_pred_fc[class_cnt],
                            doa_pred_fc[class_cnt + params['unique_classes']],
                            doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                            dist_pred_fc[class_cnt],
                        ])

                    elif flag_2sim0:
                        if sed_pred1[frame_cnt][class_cnt] > 0.5:
                            output_dict[frame_cnt].append([
                                class_cnt,
                                doa_pred1[frame_cnt][class_cnt],
                                doa_pred1[frame_cnt][class_cnt + params['unique_classes']],
                                doa_pred1[frame_cnt][class_cnt + 2 * params['unique_classes']],
                                dist_pred1[frame_cnt][class_cnt],
                            ])
                        doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                        dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                        output_dict[frame_cnt].append([
                            class_cnt,
                            doa_pred_fc[class_cnt],
                            doa_pred_fc[class_cnt + params['unique_classes']],
                            doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                            dist_pred_fc[class_cnt],
                        ])

                else:
                    output_dict.setdefault(frame_cnt, [])
                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                    dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                    output_dict[frame_cnt].append([
                        class_cnt,
                        doa_pred_fc[class_cnt],
                        doa_pred_fc[class_cnt + params['unique_classes']],
                        doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                        dist_pred_fc[class_cnt],
                    ])
    else:
        sed_pred, doa_pred = get_accdoa_labels(output_np, params['unique_classes'])
        sed_pred = reshape_3Dto2D(sed_pred)
        doa_pred = reshape_3Dto2D(doa_pred)
        for frame_cnt in range(sed_pred.shape[0]):
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt] > 0.5:
                    output_dict.setdefault(frame_cnt, []).append([
                        class_cnt,
                        doa_pred[frame_cnt][class_cnt],
                        doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                        doa_pred[frame_cnt][class_cnt + 2 * params['unique_classes']],
                    ])

    return output_dict


def _evaluate_results(params, output_folder):
    from cls_compute_seld_results import ComputeSELDResults

    score_obj = ComputeSELDResults(params)
    if params.get('evaluate_distance', False):
        ER, F, AngE, DistE, RelDistE, LR, seld_scr, _ = score_obj.get_SELD_Results(output_folder, is_jackknife=False)
        print('SELD score: {:.4f}'.format(seld_scr))
        print('F-score: {:.4f}, DOA error: {:.4f}, DistE: {:.4f}, RelDistE: {:.4f}'.format(F, AngE, DistE, RelDistE))
    else:
        ER, F, AngE, LR, seld_scr, _ = score_obj.get_SELD_Results(output_folder, is_jackknife=False)
        print('SELD score: {:.4f}'.format(seld_scr))
        print('F-score: {:.4f}, DOA error: {:.4f}'.format(F, AngE))


def main():
    parser = argparse.ArgumentParser(description='ATI two-stage SELD inference (L3 small -> conditional L4).')
    parser.add_argument('--task-id-l3', type=str, default='41')
    parser.add_argument('--job-id-l3', type=str, default='1')
    parser.add_argument('--task-id-l4', type=str, default='3')
    parser.add_argument('--job-id-l4', type=str, default='1')
    parser.add_argument('--split-idx', type=int, default=0)
    parser.add_argument('--uncertainty-threshold', type=float, default=0.08)
    parser.add_argument('--ambiguous-low', type=float, default=0.45)
    parser.add_argument('--ambiguous-high', type=float, default=0.55)
    parser.add_argument('--model-l3', type=str, default='')
    parser.add_argument('--model-l4', type=str, default='')
    parser.add_argument('--output-tag', type=str, default='ati')
    parser.add_argument('--skip-score', action='store_true')
    args = parser.parse_args()

    params_l3 = parameters.get_params(args.task_id_l3)
    params_l4 = parameters.get_params(args.task_id_l4)

    if params_l3['dataset'] != 'foa' or params_l4['dataset'] != 'foa':
        raise ValueError('ATI script currently supports FOA only. Got L3={}, L4={}'.format(params_l3['dataset'], params_l4['dataset']))
    if params_l3['modality'] != 'audio' or params_l4['modality'] != 'audio':
        raise ValueError('ATI script currently supports audio-only mode.')
    if params_l3['multi_accdoa'] != params_l4['multi_accdoa']:
        raise ValueError('L3/L4 must use same output format (multi_accdoa flag).')
    if params_l3['unique_classes'] != params_l4['unique_classes']:
        raise ValueError('L3/L4 class count mismatch.')

    split = _resolve_split(params_l3, args.split_idx)
    if params_l3['mode'] != params_l4['mode']:
        raise ValueError('L3/L4 mode mismatch: {} vs {}'.format(params_l3['mode'], params_l4['mode']))

    feat_cls_l3 = cls_feature_class.FeatureClass(params_l3, is_eval=(params_l3['mode'] == 'eval'))
    feat_cls_l4 = cls_feature_class.FeatureClass(params_l4, is_eval=(params_l4['mode'] == 'eval'))

    feat_dir_l3 = feat_cls_l3.get_normalized_feat_dir()
    feat_dir_l4 = feat_cls_l4.get_normalized_feat_dir()

    file_list_l3 = _collect_file_list(feat_dir_l3, params_l3['mode'], split)
    file_list_l4 = _collect_file_list(feat_dir_l4, params_l4['mode'], split)

    set_l4 = set(file_list_l4)
    missing_l4 = [f for f in file_list_l3 if f not in set_l4]
    if missing_l4:
        raise ValueError('L4 feature files missing for {} files, e.g. {}'.format(len(missing_l4), missing_l4[:5]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_shape_l3 = (1, 4, params_l3['feature_sequence_length'], feat_cls_l3.get_nb_mel_bins())
    out_shape_l3 = (1, params_l3['label_sequence_length'], params_l3['unique_classes'] * 3 * 4)
    model_l3 = seldnet_model.SeldModel(in_shape_l3, out_shape_l3, params_l3).to(device)

    in_shape_l4 = (1, 4, params_l4['feature_sequence_length'], feat_cls_l4.get_nb_mel_bins())
    out_shape_l4 = (1, params_l4['label_sequence_length'], params_l4['unique_classes'] * 3 * 4)
    model_l4 = seldnet_model.SeldModel(in_shape_l4, out_shape_l4, params_l4).to(device)

    model_path_l3 = _build_model_path(params_l3, args.task_id_l3, args.job_id_l3, args.split_idx, args.model_l3)
    model_path_l4 = _build_model_path(params_l4, args.task_id_l4, args.job_id_l4, args.split_idx, args.model_l4)

    if not os.path.isfile(model_path_l3):
        raise FileNotFoundError('L3 checkpoint not found: {}'.format(model_path_l3))
    if not os.path.isfile(model_path_l4):
        raise FileNotFoundError('L4 checkpoint not found: {}'.format(model_path_l4))

    model_l3.load_state_dict(torch.load(model_path_l3, map_location='cpu'))
    model_l4.load_state_dict(torch.load(model_path_l4, map_location='cpu'))
    model_l3.eval()
    model_l4.eval()

    tag = '{}_l3{}-{}_l4{}-{}_split{}_thr{:.3f}'.format(
        args.output_tag,
        args.task_id_l3,
        args.job_id_l3,
        args.task_id_l4,
        args.job_id_l4,
        args.split_idx,
        args.uncertainty_threshold,
    )

    split_tag = 'eval' if params_l3['mode'] == 'eval' else 'test'
    out_dir = os.path.join(
        params_l3['dcase_output_dir'],
        '{}_{}_{}_{}'.format(tag, _get_loc_output(params_l3), _get_loc_feat(params_l3), strftime('%Y%m%d%H%M%S', gmtime())),
    )
    if split_tag:
        out_dir = '{}_{}'.format(out_dir, split_tag)

    cls_feature_class.delete_and_create_folder(out_dir)
    print('ATI output folder: {}'.format(out_dir))

    escalated = 0
    scores = []

    for file_cnt, fname in enumerate(file_list_l3):
        feat_l3 = _load_feature_sequences(
            os.path.join(feat_dir_l3, fname),
            params_l3['feature_sequence_length'],
            feat_cls_l3.get_nb_mel_bins(),
        )
        out_l3 = _run_model(model_l3, feat_l3, device)
        score = _compute_uncertainty(
            out_l3,
            nb_classes=params_l3['unique_classes'],
            multi_accdoa=params_l3['multi_accdoa'],
            low=args.ambiguous_low,
            high=args.ambiguous_high,
        )
        scores.append(score)

        use_l4 = score > args.uncertainty_threshold
        if use_l4:
            escalated += 1
            feat_l4 = _load_feature_sequences(
                os.path.join(feat_dir_l4, fname),
                params_l4['feature_sequence_length'],
                feat_cls_l4.get_nb_mel_bins(),
            )
            out_final = _run_model(model_l4, feat_l4, device)
            final_params = params_l4
            model_tag = 'L4'
        else:
            out_final = out_l3
            final_params = params_l3
            model_tag = 'L3'

        out_dict = _output_to_dict(out_final, final_params)
        out_csv = os.path.join(out_dir, fname.replace('.npy', '.csv'))
        feat_cls_l3.write_output_format_file(out_csv, out_dict)
        print('{:04d}/{:04d} {} uncertainty={:.4f} -> {}'.format(file_cnt + 1, len(file_list_l3), fname, score, model_tag))

    esc_rate = escalated / float(max(1, len(file_list_l3)))
    print('Escalation: {}/{} ({:.2%}), mean uncertainty={:.4f}'.format(escalated, len(file_list_l3), esc_rate, float(np.mean(scores) if scores else 0.0)))

    if params_l3['mode'] == 'dev' and not args.skip_score:
        _evaluate_results(params_l3, out_dir)


if __name__ == '__main__':
    main()
