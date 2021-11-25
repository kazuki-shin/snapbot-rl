runs/AntPush_16-14-23-09/nn/AntPush.pth
python rlg_train.py --task AntPush --headless
python rlg_train.py --task AntPush --play --checkpoint runs/AntPush_16-13-57-30/nn/AntPush.pth --num_envs 100 --headless
python rlg_train.py --task Ant --play --checkpoint runs/Ant_16-17-47-53/nn/Ant.pth --num_envs 10