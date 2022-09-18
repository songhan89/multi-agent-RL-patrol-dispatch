python main.py --sectors EFL --model APPO --verbose 1 --num_workers 9 --reward_policy end_of_episode --max_iter 8000 --resume False --benchmark True --num_scenario 500 --policy multi
python main.py --sectors EFL --model APPO --verbose 1 --num_workers 11 --reward_policy end_of_episode --max_iter 8000 --resume False --benchmark True --num_scenario 500 --policy single
python main.py --sectors EFL --model A3C --verbose 1 --num_workers 11 --reward_policy end_of_episode --max_iter 8000 --resume False --benchmark True --num_scenario 500 --policy single
python main.py --sectors EFL --model A3C --verbose 1 --num_workers 9 --reward_policy end_of_episode --max_iter 8000 --resume False --benchmark True --num_scenario 500 --policy multi
#python main.py --sectors EFL --model PPO --verbose 1 --num_workers 11 --reward_policy end_of_episode --max_iter 500 --resume False --benchmark True --num_scenario 500
