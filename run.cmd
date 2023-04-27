::python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy stepwise --max_iter 1000 --resume False --benchmark True --num_scenario 500 --policy multi --exploration Greedy
::python main.py --sectors EFL --model APPO --verbose 1 --num_workers 3 --reward_policy end_of_episode --max_iter 1000 --resume False --benchmark True --num_scenario 500 --policy multi --exploration EpsilonGreedy --nn_network 128x128_relu_attention
::python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100 --resume False --benchmark True --num_scenario 500 --policy multi --exploration EpsilonGreedy
::python main.py --sectors EFL --model A3C --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100 --resume False --benchmark True --num_scenario 500 --policy multi --exploration EpsilonGreedy


python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100
--resume False --benchmark True --num_scenario 500 --policy multi --theta_step 1.0 --exploration EpsilonGreedy
python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100
--resume False --benchmark True --num_scenario 500 --policy multi --theta_step 0.5 --exploration EpsilonGreedy
python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100
--resume False --benchmark True --num_scenario 500 --policy multi --theta_step 0.2 --exploration EpsilonGreedy
python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100
--resume False --benchmark True --num_scenario 500 --policy multi --theta_step 0.1 --exploration EpsilonGreedy
python main.py --sectors EFL --model APPO --verbose 1 --num_workers 5 --reward_policy end_of_episode --max_iter 100
--resume False --benchmark True --num_scenario 500 --policy multi --theta_step 0 --exploration EpsilonGreedy

