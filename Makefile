# SHELL := /usr/bin/env bash

IMAGE := pypolsar
VERSION := latest
POETRY ?= $(HOME)/.conda/envs/mans_is/bin/poetry


#! An ugly hack to create individual flags
ifeq ($(STRICT), 1)
	POETRY_COMMAND_FLAG =
	PIP_COMMAND_FLAG =
	SAFETY_COMMAND_FLAG =
	BANDIT_COMMAND_FLAG =
	SECRETS_COMMAND_FLAG =
	BLACK_COMMAND_FLAG =
	DARGLINT_COMMAND_FLAG =
	ISORT_COMMAND_FLAG =
	MYPY_COMMAND_FLAG =
else
	POETRY_COMMAND_FLAG = -
	PIP_COMMAND_FLAG = -
	SAFETY_COMMAND_FLAG = -
	BANDIT_COMMAND_FLAG = -
	SECRETS_COMMAND_FLAG = -
	BLACK_COMMAND_FLAG = -
	DARGLINT_COMMAND_FLAG = -
	ISORT_COMMAND_FLAG = -
	MYPY_COMMAND_FLAG = -
endif

##! Please tell me how to use `for loops` to create variables in Makefile :(
##! If you have better idea, please PR me in https://github.com/TezRomacH/python-package-template

ifeq ($(POETRY_STRICT), 1)
	POETRY_COMMAND_FLAG =
else ifeq ($(POETRY_STRICT), 0)
	POETRY_COMMAND_FLAG = -
endif

ifeq ($(PIP_STRICT), 1)
	PIP_COMMAND_FLAG =
else ifeq ($(PIP_STRICT), 0)
	PIP_COMMAND_FLAG = -
endif

ifeq ($(SAFETY_STRICT), 1)
	SAFETY_COMMAND_FLAG =
else ifeq ($SAFETY_STRICT), 0)
	SAFETY_COMMAND_FLAG = -
endif

ifeq ($(BANDIT_STRICT), 1)
	BANDIT_COMMAND_FLAG =
else ifeq ($(BANDIT_STRICT), 0)
	BANDIT_COMMAND_FLAG = -
endif

ifeq ($(SECRETS_STRICT), 1)
	SECRETS_COMMAND_FLAG =
else ifeq ($(SECRETS_STRICT), 0)
	SECRETS_COMMAND_FLAG = -
endif

ifeq ($(BLACK_STRICT), 1)
	BLACK_COMMAND_FLAG =
else ifeq ($(BLACK_STRICT), 0)
	BLACK_COMMAND_FLAG = -
endif

ifeq ($(DARGLINT_STRICT), 1)
	DARGLINT_COMMAND_FLAG =
else ifeq (DARGLINT_STRICT), 0)
	DARGLINT_COMMAND_FLAG = -
endif

ifeq ($(ISORT_STRICT), 1)
	ISORT_COMMAND_FLAG =
else ifeq ($(ISORT_STRICT), 0)
	ISORT_COMMAND_FLAG = -
endif


ifeq ($(MYPY_STRICT), 1)
	MYPY_COMMAND_FLAG =
else ifeq ($(MYPY_STRICT), 0)
	MYPY_COMMAND_FLAG = -
endif

#! The end of the ugly part. I'm really sorry

.PHONY: download-poetry
download-poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

.PHONY: install
install:
	poetry lock -n
	poetry install -n
ifneq ($(NO_PRE_COMMIT), 1)
	poetry run pre-commit install
endif

.PHONY: check-safety
check-safety:
	$(POETRY_COMMAND_FLAG)poetry check
	$(PIP_COMMAND_FLAG)poetry run pip check
	$(SAFETY_COMMAND_FLAG)poetry run safety check --full-report
	$(BANDIT_COMMAND_FLAG)poetry run bandit -ll -r pypolsar/

.PHONY: check-style
check-style:
	$(BLACK_COMMAND_FLAG)poetry run black --config pyproject.toml --diff --check ./
	$(DARGLINT_COMMAND_FLAG)poetry run darglint -v 2 **/*.py
	$(ISORT_COMMAND_FLAG)poetry run isort --settings-path pyproject.toml --check-only
	$(MYPY_COMMAND_FLAG)poetry run mypy --config-file setup.cfg pypolsar tests/**/*.py

.PHONY: lint-black
lint-black:
	@echo "\033[92m< linting using black...\033[0m"
	@$(POETRY) run black .
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-flake8
lint-flake8:
	@echo "\033[92m< linting using flake8...\033[0m"
	@$(POETRY) run flake8 aioaerospike tests
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-isort
lint-isort:
	@echo "\033[92m< linting using isort...\033[0m"
	@$(POETRY) run isort --recursive .
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-mypy
lint-mypy:
	@echo "\033[92m< linting using mypy...\033[0m"
	@$(POETRY) run mypy pypolsar tests/**/*.py
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint
lint: lint-black lint-flake8 lint-isort lint-mypy

.PHONY: lint-check-black
lint-check-black:
	@echo "\033[92m< linting using black...\033[0m"
	@$(POETRY) run black --check --diff .
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-check-flake8
lint-check-flake8:
	@echo "\033[92m< linting using flake8...\033[0m"
	@$(POETRY) run flake8 aioaerospike tests
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-check-isort
lint-check-isort:
	@echo "\033[92m< linting using isort...\033[0m"
	@$(POETRY) run isort --check-only --diff --recursive .
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-check-mypy
lint-check-mypy:
	@echo "\033[92m< linting using mypy...\033[0m"
	@$(POETRY) run mypy --ignore-missing-imports --follow-imports=silent aioaerospike tests
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-check
lint-check: lint-check-black lint-check-flake8 lint-check-isort lint-check-mypy

.PHONY: major
major:
	poetry version major

.PHONY: minor
minor:
	poetry version minor

.PHONY: patch
patch:
	poetry version patch

.PHONY: premajor
premajor:
	poetry version premajor

.PHONY: preminor
preminor:
	poetry version preminor

.PHONY: prepatch
prepatch:
	poetry version prepatch

.PHONY: prerelease
prerelease:
	poetry version prerelease


.PHONY: publish
publish:
	poetry build && poetry install && poetry publish

.PHONY: publish-patch
publish-patch: lint patch publish

#	major	1.3.0	2.0.0
#	minor	2.1.4	2.2.0
#	patch	4.1.1	4.1.2
#	premajor	1.0.2	2.0.0-alpha.0
#	preminor	1.0.2	1.1.0-alpha.0
#	prepatch	1.0.2	1.0.3-alpha.0
#	prerelease	1.0.2	1.0.3-alpha.0
#	prerelease	1.0.3-alpha.0	1.0.3-alpha.1
#	prerelease	1.0.3-beta.0	1.0.3-beta.1

.PHONY: codestyle
codestyle:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	poetry run pytest

.PHONY: lint-test
lint-test: test check-safety check-style

# Example: make docker VERSION=latest
# Example: make docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker
docker:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make clean_docker VERSION=latest
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: clean_docker
clean_docker:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

.PHONY: clean_build
clean:
	rm -rf build/

.PHONY: clean
clean: clean_build clean_docker
