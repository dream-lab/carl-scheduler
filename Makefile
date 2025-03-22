run-ff:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo ffb --state MonitoringLogs/ffb_cost 2>&1 | tee ../SimulatorLogs/ffb_cost.log

run-bfb:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo bfb --state MonitoringLogs/bfb_cost 2>&1 | tee ../SimulatorLogs/bfb_cost.log

run-vpb:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo vpb --state MonitoringLogs/vpb_cost 2>&1 | tee ../SimulatorLogs/vpb_cost.log

run-bc:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo bc --state MonitoringLogs/bc_cost 2>&1 | tee ../SimulatorLogs/bc_cost.log
run-ppo:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo ppo --state MonitoringLogs/ppo_cost 2>&1 | tee ../SimulatorLogs/ppo_cost.log

run-gail:
	python -u main.py --mf ../Data/vm.csv --cf ../Data/container.csv --algo gail --state MonitoringLogs/gail_cost 2>&1 | tee ../SimulatorLogs/gail_cost.log