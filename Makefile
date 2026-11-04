.PHONY: install test reproduce medical-latent latent-test transition-diagnosis \
	temporal-jepa trajectories temporal-jepa-ablation clean

PYTHON := python3
RUN := MPLBACKEND=Agg PYTHONPATH=src $(PYTHON)

install:
	$(PYTHON) -m pip install -e '.[dev,torch]'

test:
	$(PYTHON) -m pytest -q

# Full paper pipeline. latent-test creates the checkpoints consumed by
# transition-diagnosis; temporal-jepa creates those consumed by trajectories.
reproduce: test medical-latent latent-test transition-diagnosis temporal-jepa trajectories temporal-jepa-ablation

medical-latent:
	$(RUN) scripts/run_medical_latent.py

latent-test:
	$(RUN) scripts/run_latent_test.py

transition-diagnosis: latent-test
	$(RUN) scripts/run_transition_diagnosis.py

temporal-jepa:
	$(RUN) scripts/run_temporal_jepa.py

trajectories: temporal-jepa
	$(RUN) scripts/evaluate_interventional_trajectories.py

temporal-jepa-ablation:
	$(RUN) scripts/run_temporal_jepa_ablation.py

clean:
	find . -type d -name '__pycache__' -prune -exec find {} -depth -delete \;
	find . -type d -name '.pytest_cache' -prune -exec find {} -depth -delete \;
	rm -rf results figures
