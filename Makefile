.PHONY: install test core robustness combination two-stage multistep validation transformer medical-part1 medical-latent latent-diagnosis functional-use latent-test transition-diagnosis compositional-latent closure-training temporal-jepa temporal-jepa-ablation synthetic clean

install:
	python3 -m pip install -e .

test:
	python3 -m pytest -q

core:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_core_experiment.py

robustness:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_robustness.py

combination:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_combination.py

two-stage:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_two_stage.py

multistep:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_multistep.py

validation:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_validation_suite.py

transformer:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/train_small_transformer.py

medical-part1:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_medical_part1.py

medical-latent:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_medical_latent.py

latent-diagnosis:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_latent_failure_diagnosis.py

functional-use:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_functional_use.py

latent-test:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_latent_test.py

transition-diagnosis:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_transition_diagnosis.py

compositional-latent:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_compositional_latent.py

closure-training:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_closure_training.py

temporal-jepa:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_temporal_jepa.py

temporal-jepa-ablation:
	MPLBACKEND=Agg PYTHONPATH=src python3 scripts/run_temporal_jepa_ablation.py

synthetic:
	PYTHONPATH=src python3 scripts/run_synthetic.py

clean:
	rm -rf results figures .pytest_cache
