#!/usr/bin/env python3
import argparse
import os
from time import gmtime, strftime

import torch
import torch.nn as nn

import parameters
import cls_data_generator
import cls_feature_class
import seldnet_model
import train_seldnet
from cls_compute_seld_results import ComputeSELDResults


def infer_test_split(dataset_dir):
    if "2024" in dataset_dir:
        return [4]
    if "2023" in dataset_dir:
        return [4]
    if "2022" in dataset_dir:
        return [4]
    if "2021" in dataset_dir:
        return [6]
    if "2020" in dataset_dir:
        return [1]
    raise RuntimeError("Unknown dataset split from dataset_dir={}".format(dataset_dir))


def main():
    parser = argparse.ArgumentParser(description="Best 모델로 test-only 평가")
    parser.add_argument("task_id", help="parameters.py task-id (예: 33)")
    parser.add_argument("job_id", help="학습시 사용한 job-id (예: decoupled33_0301_1539)")
    args = parser.parse_args()

    task_id = str(args.task_id)
    job_id = str(args.job_id)

    params = parameters.get_params(task_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_split = infer_test_split(params["dataset_dir"])

    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=test_split, shuffle=False, per_file=True
    )

    if params["modality"] == "audio_visual":
        data_in, vid_data_in, data_out = data_gen_test.get_data_sizes()
        model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
    else:
        data_in, data_out = data_gen_test.get_data_sizes()
        model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

    loc_feat = params["dataset"]
    if params["dataset"] == "mic":
        if params["use_salsalite"]:
            loc_feat = "{}_salsa".format(params["dataset"])
        else:
            loc_feat = "{}_gcc".format(params["dataset"])
    loc_output = "multiaccdoa" if params["multi_accdoa"] else "accdoa"
    unique_name = "{}_{}_{}_split0_{}_{}".format(
        task_id, job_id, params["mode"], loc_output, loc_feat
    )

    model_path = os.path.join(params["model_dir"], "{}_model.h5".format(unique_name))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    criterion = seldnet_model.MSELoss_ADPIT() if params["multi_accdoa"] else nn.MSELoss()

    out_dir = os.path.join(
        params["dcase_output_dir"],
        "{}_{}_testonly".format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())),
    )
    cls_feature_class.delete_and_create_folder(out_dir)

    test_loss = train_seldnet.test_epoch(data_gen_test, model, criterion, out_dir, params, device)
    score_obj = ComputeSELDResults(params)
    ER, F, LE, DistE, RelDistE, LR, seld_scr, _ = score_obj.get_SELD_Results(out_dir)

    os.makedirs("logs", exist_ok=True)
    summary_path = os.path.join(
        "logs",
        "eval_task{}_{}_{}_summary.txt".format(task_id, job_id, strftime("%Y%m%d_%H%M%S", gmtime())),
    )

    summary_text = "\n".join(
        [
            "=== Eval Test-Only Summary ===",
            "완료시각: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
            "task_id: {}".format(task_id),
            "job_id: {}".format(job_id),
            "model_path: {}".format(model_path),
            "output_dir: {}".format(out_dir),
            "",
            "[최종 지표]",
            "test_loss={:.4f}".format(test_loss),
            "SELD={:.4f}".format(seld_scr),
            "F-score={:.4f}".format(F),
            "AngularError={:.4f}".format(LE),
            "DistanceError={:.4f}".format(DistE),
            "RelativeDistanceError={:.4f}".format(RelDistE),
            "ER={:.4f}".format(ER),
            "LR={:.4f}".format(LR),
        ]
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print("output_dir:", out_dir)
    print("test_loss={:.4f}".format(test_loss))
    print(
        "SELD={:.4f}, F={:.4f}, AE={:.4f}, DistE={:.4f}, RelDistE={:.4f}".format(
            seld_scr, F, LE, DistE, RelDistE
        )
    )
    print("summary_file:", summary_path)


if __name__ == "__main__":
    main()
