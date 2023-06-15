# python vap/train_b2.py --vap_load_pretrained 0\
#                     --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_hkust_31.csv'\
#                     --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_hkust_31.csv'\
#                     --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_hkust_729.csv'\
#                     --vap_freeze_encoder 1\
#                     --data_batch_size 2\
                    #--fast_dev_run 100\
                    #--data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_hkust_31.csv'\
# python vap/train_b2.py --vap_load_pretrained 1\
#                     --vap_freeze_encoder 1\
#                     --data_batch_size 2\
#                     --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_switchboard_11295.csv'\
#                     --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_switchboard_24161.csv'\
#                     --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_switchboard_11479.csv'\

## change val/train to be the same with chinese
python vap/train_b2.py --vap_load_pretrained 1\
                    --vap_freeze_encoder 1\
                    --data_batch_size 2\
                    --data_val_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/val_hkust_2684.csv'\
                    --data_train_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/train_hkust_24132.csv'\
                    --data_test_path '../vap_dataset/data/sliding_window_ad20_ov1_ho2/test_hkust_729.csv'\
                    #--fast_dev_run 10
             #       --opt_find_learning_rate
# python vap/train_b2.py --vap_load_pretrained 1\
#                     --vap_freeze_encoder 1\
#                     --data_batch_size 2\
#                     --data_val_path '../vap_dataset/data/sliding_window_ad21_ov1_ho2/val_hkust_2522.csv'\
#                     --data_train_path '../vap_dataset/data/sliding_window_ad21_ov1_ho2/train_hkust_22625.csv'\
#                     --data_test_path '../vap_dataset/data/sliding_window_ad21_ov1_ho2/test_hkust_685.csv'\
#                     --fast_dev_run 10