# python vap/evaluation.py --vap_freeze_encoder 0\
#         --data_batch_size 8\
#         --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_switchboard_11295.csv'\
#         --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_switchboard_24161.csv'\
#         --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_switchboard_11479.csv'\
#         --checkpoint example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt
 #--checkpoint runs/VapGPT/3knpg1n7/checkpoints/VapGPT_50Hz_ad20s_134-epoch14-val_2.55.ckpt\
# python vap/evaluation.py --vap_freeze_encoder 0\
#         --data_batch_size 8\
#         --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_hkust_24132.csv'\
#         --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_hkust_2684.csv'\
#         --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_hkust_729.csv'\
#         --checkpoint "example/VapGPT_50Hz_ad20s_134-epoch24-val_3.01.ckpt"



## best mandarin: 
# runs/VapGPT/369338ud/checkpoints/VapGPT_50Hz_ad20s_134-epoch7-val_2.32.ckpt


# english
# python vap/evaluation.py --vap_freeze_encoder 1\
#                     --data_batch_size 2\
#                     --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_switchboard_2619.csv'\
#                     --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_switchboard_23556.csv'\
#                     --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_switchboard_11479.csv'\
#                         --checkpoint runs/VapGPT/VapGPT_en130h_pbody-epoch11-val_2.12.ckpt

python vap/evaluation.py --vap_freeze_encoder 1\
                    --data_batch_size 2\
                   --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_hkust_24132.csv'\
                     --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_hkust_2684.csv'\
                     --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_hkust_729.csv'\
    --checkpoint runs/VapGPT/369338ud/checkpoints/VapGPT_50Hz_ad20s_134-epoch7-val_2.32.ckpt
