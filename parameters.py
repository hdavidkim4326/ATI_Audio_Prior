# SELDnet의 특징 추출, 모델, 학습 관련 파라미터를 이 파일에서 변경할 수 있습니다.
#
# 기본 파라미터는 가급적 변경하지 말고, 아래 if-else처럼 고유한 <task-id> 케이스를 추가해 사용하세요.
# 이렇게 하면 이후에도 동일 설정을 쉽게 재현할 수 있습니다.
import os

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### 기본 파라미터 ##############
    params = dict(
        quick_test=True,  # 빠른 점검용 모드(작은 데이터/짧은 epoch로 학습/평가)

        finetune_mode=True,  # 기존 모델 가중치로 파인튜닝(pretrained_model_weights 경로 필요)
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        # 입력 경로
        # dataset_dir='DCASE2020_SELD_dataset/',  # foa/mic 및 메타데이터 폴더를 포함한 루트 경로
        dataset_dir='../DCASE2024_SELD_dataset/',

        # 출력 경로
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # 추출된 특징/라벨 저장 경로
        feat_label_dir='../DCASE2024_SELD_dataset/seld_feat_label/',

        model_dir='models',  # 학습된 모델과 학습 곡선을 저장하는 폴더
        dcase_output_dir='results',  # 녹음 파일 단위 결과를 저장하는 경로

        # 데이터셋 로딩 파라미터
        mode='dev',  # 'dev': 개발셋, 'eval': 평가셋
        dataset='foa',  # 'foa': Ambisonic, 'mic': 마이크 신호

        # 특징 추출 파라미터
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # MIC 데이터셋에서만 사용. True면 SALSA-lite, 아니면 GCC
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # L2 입력 제어 파라미터(FOA 전용)
        l2_enable=False,
        l2_mode='foa_tf_softmask',
        nb_channels=4,  # FOA 기본 채널(WXYZ). task36에서는 saliency 채널 추가로 5 사용
        l2_mask_tau=0.05,
        l2_mask_tau_base=0.2,
        l2_mask_k=10.0,
        l2_diffuseness_alpha=0.5,
        l2_iv_blend_lambda=0.5,
        l2_iv_blend_lambda_min=0.2,
        l2_iv_blend_lambda_max=0.6,
        l2_iv_blend_gamma=1.5,
        # Task36: L3-guided spatial saliency 채널 생성 파라미터
        l3_prior_dir=os.environ.get('L3_PRIOR_DIR', ''),  # 비어 있으면 results_audio 내 task3 결과 자동 탐색
        l3_prior_topk=3,                # 프레임별 L3 pseudo 방향 top-k 사용
        l3_saliency_beta=12.0,          # sigmoid 기울기
        l3_saliency_tau=0.15,           # 방향 정렬도 임계값
        l3_saliency_default=0.5,        # prior 부재 시 기본 saliency
        l3_saliency_temp_smooth=0.8,    # 시간축 EMA 강도(0이면 비활성)
        l3_saliency_use_energy_gate=True,
        l3_saliency_energy_gamma=0.5,
        # Task37: L3-guided active IV reinforcement
        l3_iv_gain_max=0.35,            # XYZ 능동 강조 최대 증폭(1 + gain)
        l3_iv_delta_clip=0.30,          # IV residual 안정화를 위한 클리핑
        l3_iv_conf_gamma=1.2,           # prior confidence 비선형 지수
        l3_iv_conf_floor=0.05,          # prior confidence 하한(0~1)

        # 모델 타입
        modality='audio',  # 'audio' 또는 'audio_visual'
        multi_accdoa=False,  # False: Single-ACCDOA, True: Multi-ACCDOA
        thresh_unify=15,    # Multi-ACCDOA 전용 추론 통합 각도 임계값(도 단위)
        accdoa_sed_thresh=0.5,  # ACCDOA activity threshold (기본 0.5)

        # DNN 모델 파라미터
        label_sequence_length=50,    # 특징 시퀀스 길이
        batch_size=128,              # 배치 크기
        dropout_rate=0.05,           # 모든 레이어에 공통으로 적용되는 드롭아웃 비율
        nb_cnn2d_filt=64,           # CNN 필터 수(레이어 공통)
        f_pool_size=[4, 4, 2],      # CNN 주파수 풀링(리스트 길이=레이어 수, 값=레이어별 풀링)

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN 크기

        nb_epochs=250,  # 최대 학습 epoch
        lr=1e-3,
        early_stop_metric='f_score',  # 'f_score' 또는 'seld_score'
        early_stop_patience=None,     # None이면 nb_epochs 기반으로 동작
        early_stop_min_delta=0.0,     # 개선으로 인정할 최소 변화량
        early_stop_seld_weight=0.35,  # early_stop_metric='f_seld_hybrid'일 때 사용

        # 평가 지표
        average='macro',                 # 'micro': 샘플 단위 평균, 'macro': 클래스 단위 평균
        segment_based_metrics=False,     # True면 segment 기반, False면 frame 기반 지표
        evaluate_distance=True,          # True면 거리 오차 계산 및 거리 임계값 적용
        lad_doa_thresh=20,               # 검출 지표 계산용 DOA 오차 임계값
        lad_dist_thresh=float('inf'),    # 검출 지표 계산용 절대 거리 오차 임계값
        lad_reldist_thresh=float('1'),  # 검출 지표 계산용 상대 거리 오차 임계값
    )

    # ########### 사용자 정의 파라미터 ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '31':
        print("FOA + multi ACCDOA + L2 softmask\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True

    elif argv == '32':
        print("FOA + multi ACCDOA + L2 Adaptive Softmask\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_tf_adaptive_softmask'
        params['l2_mask_tau_base'] = 0.2
        params['l2_mask_k'] = 10.0
        params['l2_diffuseness_alpha'] = 0.5

    elif argv == '33':
        print("FOA + multi ACCDOA + L2 Decoupled Adaptive Softmask\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_tf_decoupled_adaptive_softmask'
        params['l2_mask_tau_base'] = 0.2
        params['l2_mask_k'] = 10.0
        params['l2_diffuseness_alpha'] = 0.5

    elif argv == '34':
        print("FOA + multi ACCDOA + L2 IV-Residual Adaptive\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_tf_iv_adaptive_residual'
        params['l2_mask_tau_base'] = 0.2
        params['l2_mask_k'] = 10.0
        params['l2_diffuseness_alpha'] = 0.5
        params['l2_iv_blend_lambda'] = 0.5

    elif argv == '35':
        print("FOA + multi ACCDOA + L2 IV-Residual Adaptive Dynamic + Early Stop\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_tf_iv_adaptive_residual_dynamic'
        params['l2_mask_tau_base'] = 0.2
        params['l2_mask_k'] = 10.0
        params['l2_diffuseness_alpha'] = 0.5
        params['l2_iv_blend_lambda_min'] = 0.15
        params['l2_iv_blend_lambda_max'] = 0.55
        params['l2_iv_blend_gamma'] = 1.5

        # 성능 정체가 길어지면 조기 종료 후 best 모델로 자동 test 수행
        params['early_stop_metric'] = 'seld_score'
        params['early_stop_patience'] = 40
        params['early_stop_min_delta'] = 0.001

    elif argv == '36':
        print("FOA + multi ACCDOA + L3-Guided Spatial Saliency Augmentation (5ch)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_l3_guided_reinforcement'
        params['nb_channels'] = 5

        # task3 결과 기반 pseudo-label prior를 사용해 saliency 채널 생성
        # 기본값은 환경변수 L3_PRIOR_DIR를 따른다. 비어 있으면 자동 탐색.
        params['l3_prior_dir'] = os.environ.get('L3_PRIOR_DIR', params.get('l3_prior_dir', ''))
        params['l3_prior_topk'] = 3
        params['l3_saliency_beta'] = 12.0
        params['l3_saliency_tau'] = 0.15
        params['l3_saliency_default'] = 0.5
        params['l3_saliency_temp_smooth'] = 0.8
        params['l3_saliency_use_energy_gate'] = True
        params['l3_saliency_energy_gamma'] = 0.5

        # 성능 정체 시 조기 종료 후 best 모델 자동 test
        params['early_stop_metric'] = 'seld_score'
        params['early_stop_patience'] = 40
        params['early_stop_min_delta'] = 0.001

    elif argv == '37':
        print("FOA + multi ACCDOA + L3-Guided Active IV Reinforcement (7ch)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['l2_enable'] = True
        params['l2_mode'] = 'foa_tf_iv_l3_aware_residual_dynamic'
        params['nb_channels'] = 4  # 입력 채널 구조는 baseline(7ch feature) 유지

        # adaptive IV residual 기본 파라미터
        params['l2_mask_tau_base'] = 0.2
        params['l2_mask_k'] = 10.0
        params['l2_diffuseness_alpha'] = 0.5
        params['l2_iv_blend_lambda_min'] = 0.18
        params['l2_iv_blend_lambda_max'] = 0.65
        params['l2_iv_blend_gamma'] = 1.5

        # task3 결과 기반 pseudo prior + 능동 강조 파라미터
        params['l3_prior_dir'] = os.environ.get('L3_PRIOR_DIR', params.get('l3_prior_dir', ''))
        params['l3_prior_topk'] = 3
        params['l3_saliency_beta'] = 10.0
        params['l3_saliency_tau'] = 0.10
        params['l3_saliency_default'] = 0.45
        params['l3_saliency_temp_smooth'] = 0.6
        params['l3_saliency_use_energy_gate'] = True
        params['l3_saliency_energy_gamma'] = 0.35
        params['l3_iv_gain_max'] = 0.35
        params['l3_iv_delta_clip'] = 0.30
        params['l3_iv_conf_gamma'] = 1.2
        params['l3_iv_conf_floor'] = 0.05

        # F-score와 SELD를 함께 밀어올리기 위한 조기종료 기준
        params['early_stop_metric'] = 'f_seld_hybrid'
        params['early_stop_seld_weight'] = 0.35
        params['early_stop_patience'] = 45
        params['early_stop_min_delta'] = 0.0005
        params['accdoa_sed_thresh'] = 0.48

    elif argv == '41':
        print("FOA + multi ACCDOA + SMALL L3 model\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = False

        params['nb_cnn2d_filt'] = 32
        params['nb_rnn_layers'] = 1
        params['rnn_size'] = 64
        params['nb_heads'] = 4
        params['nb_self_attn_layers'] = 1
        params['nb_transformer_layers'] = 1
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 64

    elif argv == '42':
        print("FOA + multi ACCDOA + SMALL L3 model + L2 softmask\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['l2_enable'] = True

        params['nb_cnn2d_filt'] = 32
        params['nb_rnn_layers'] = 1
        params['rnn_size'] = 64
        params['nb_heads'] = 4
        params['nb_self_attn_layers'] = 1
        params['nb_transformer_layers'] = 1
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 64

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN 시간축 풀링
    params['patience'] = int(params['nb_epochs']) if params.get('early_stop_patience') is None else int(params['early_stop_patience'])  # patience 도달 시 학습 중단
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
