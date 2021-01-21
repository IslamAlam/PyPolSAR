# PyPolSAR

<div align="center">

[![Build status](https://github.com/IslamAlam/pypolsar/workflows/build/badge.svg?branch=master&event=push)](https://github.com/IslamAlam/pypolsar/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/pypolsar.svg)](https://pypi.org/project/pypolsar/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/IslamAlam/pypolsar/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/IslamAlam/pypolsar/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/IslamAlam/pypolsar/releases)
[![License](https://img.shields.io/github/license/IslamAlam/pypolsar)](https://github.com/IslamAlam/pypolsar/blob/master/LICENSE)

PyPolSAR is a python module for Polarimetric Synthetic Aperture Radar (PolSAR) data processing.
</div>

Documentation & Software Citation

To see the latest full documentation click on [here](https://pypolsar.readthedocs.io/en/latest/).

If you use the software in a publication then please cite it using the Zenodo DOI. Be aware that this badge links to the latest package version.

Please select your specific version at [to do in future] to get the DOI of that version. You should normally always use the DOI for the specific version of your record in citations. This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
This package should be installable through pip which downloads the package from the python package repository Pypi. However, pypolsar also needs some packages that depend on C or Fortran libraries (like netCDF4). They should be installed first with conda. See http://conda.pydata.org/docs/ on how to use it. We recommend using either Anaconda or Miniconda.

```bash
conda install -c conda-forge numpy scipy pandas netCDF4 cython libgdal gdal
```
Afterwards pypolsar can be installed via pip.

```bash
 pip install pypolsar
```    
You can also install all needed (conda and pip) dependencies at once using the following commands after cloning this repository. This is recommended for developers of the package. Note that the git --recursive flag will clone test-data, which is needed by some tests.

```bash
git clone https://github.com/IslamAlam/pypolsar.git --recursive
cd pypolsar
conda create -n pypolsar python=3.6 # or any supported python version
source activate pypolsar
conda update -f environment.yml -n pypolsar
python setup.py develop
```


### Initial setting up

- Set up [Dependabot](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates) to ensure you have the latest dependencies.
- Set up [Stale bot](https://github.com/apps/stale) for automatic issue closing.

### Poetry

All manipulations with dependencies are executed through Poetry. If you're new to it, look through [the documentation](https://python-poetry.org/docs/).

<details>
<summary>Notes about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy`
- `poetry run pytest`
- `poetry build`
- etc

</p>
</details>

### Building your package

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish 🙂 `poetry publish --build`

## What's next

Well, that's up to you. I can only recommend the packages and articles that helped me.

Packages:

- [`Typer`](https://github.com/tiangolo/typer) is great for creating CLI applications.
- [`Rich`](https://github.com/willmcgugan/rich) makes it easy to add beautiful formatting in the terminal.
- [`FastAPI`](https://github.com/tiangolo/fastapi) is a type-driven asynchronous web framework.
- [`IceCream`](https://github.com/gruns/icecream) is a little library for sweet and creamy debugging

Articles:

- [Open Source Guides](https://opensource.guide/)
- [GitHub Actions Documentation](https://help.github.com/en/actions)
- Maybe you would like to add [gitmoji](https://gitmoji.carloscuesta.me/) to commit names. This is really funny. 😄

## 🚀 Features

For your development we've prepared:

- Supports for `Python 3.7` and higher.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/IslamAlam/pypolsar/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/IslamAlam/pypolsar/blob/master/setup.cfg).
- Power of [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade) formatters.
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with formatters above.
- Type checks with the configured [`mypy`](https://mypy.readthedocs.io).
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint).
- Security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit).
- Well-made [`.editorconfig`](https://github.com/IslamAlam/pypolsar/blob/master/.editorconfig), [`.dockerignore`](https://github.com/IslamAlam/pypolsar/blob/master/.dockerignore), and [`.gitignore`](https://github.com/IslamAlam/pypolsar/blob/master/.gitignore). You don't have to worry about those things.

For building and deployment:

- `GitHub` integration.
- [`Makefile`](https://github.com/IslamAlam/pypolsar/blob/master/Makefile#L89) for building routines. Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc. More details at [Makefile summary](#makefile-usage)).
- [Dockerfile](https://github.com/IslamAlam/pypolsar/blob/master/docker/Dockerfile) for your package.
- `Github Actions` with predefined [build workflow](https://github.com/IslamAlam/pypolsar/blob/master/.github/workflows/build.yml) as the default CI/CD.
- Always up-to-date dependencies with [`@dependabot`](https://dependabot.com/) (You will only [enable it](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates)).
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). It creates a list of changes based on labels in merged `Pull Requests`. You can see labels (aka `categories`) in [`release-drafter.yml`](https://github.com/IslamAlam/pypolsar/blob/master/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

For creating your open source community:

- Ready-to-use [Pull Requests templates](https://github.com/IslamAlam/pypolsar/blob/master/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/IslamAlam/pypolsar/tree/master/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [`Stale bot`](https://github.com/apps/stale) that closes abandoned issues after a period of inactivity. (You will only [need to setup free plan](https://github.com/marketplace/stale)). Configuration is [here](https://github.com/IslamAlam/pypolsar/blob/master/.github/.stale.yml).
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

## Installation

```bash
pip install pypolsar
```

or install with `Poetry`

```bash
poetry add pypolsar
```

Then you can run

```bash
pypolsar --help
```

```bash
pypolsar --name Roman
```

or if installed with `Poetry`:

```bash
poetry run pypolsar --help
```

```bash
poetry run pypolsar --name Roman
```

### Makefile usage

[`Makefile`](https://github.com/IslamAlam/pypolsar/blob/master/Makefile) contains many functions for fast assembling and convenient work.

<details>
<summary>1. Download Poetry</summary>
<p>

```bash
make download-poetry
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

```bash
make install
```

If you do not want to install pre-commit hooks, run the command with the NO_PRE_COMMIT flag:

```bash
make install NO_PRE_COMMIT=1
```

</p>
</details>

<details>
<summary>3. Check the security of your code</summary>
<p>

```bash
make check-safety
```

This command launches a `Poetry` and `Pip` integrity check as well as identifies security issues with `Safety` and `Bandit`. By default, the build will not crash if any of the items fail. But you can set `STRICT=1` for the entire build, or you can configure strictness for each item separately.

```bash
make check-safety STRICT=1
```

or only for `safety`:

```bash
make check-safety SAFETY_STRICT=1
```

multiple

```bash
make check-safety PIP_STRICT=1 SAFETY_STRICT=1
```

> List of flags for `check-safety` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`.

</p>
</details>

<details>
<summary>4. Check the codestyle</summary>
<p>

The command is similar to `check-safety` but to check the code style, obviously. It uses `Black`, `Darglint`, `Isort`, and `Mypy` inside.

```bash
make check-style
```

It may also contain the `STRICT` flag.

```bash
make check-style STRICT=1
```

> List of flags for `check-style` (can be set to `1` or `0`): `STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>5. Run all the codestyle formaters</summary>
<p>

Codestyle uses `pre-commit` hooks, so ensure you've run `make install` before.

```bash
make codestyle
```

</p>
</details>

<details>
<summary>6. Run tests</summary>
<p>

```bash
make test
```

</p>
</details>

<details>
<summary>7. Run all the linters</summary>
<p>

```bash
make lint
```

the same as:

```bash
make test && make check-safety && make check-style
```

> List of flags for `lint` (can be set to `1` or `0`): `STRICT`, `POETRY_STRICT`, `PIP_STRICT`, `SAFETY_STRICT`, `BANDIT_STRICT`, `BLACK_STRICT`, `DARGLINT_STRICT`, `ISORT_STRICT`, `MYPY_STRICT`.

</p>
</details>

<details>
<summary>8. Build docker</summary>
<p>

```bash
make docker
```

which is equivalent to:

```bash
make docker VERSION=latest
```

More information [here](https://github.com/IslamAlam/pypolsar/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup docker</summary>
<p>

```bash
make clean_docker
```

or to remove all build

```bash
make clean
```

More information [here](https://github.com/IslamAlam/pypolsar/tree/master/docker).

</p>
</details>

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/IslamAlam/pypolsar/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

For Pull Request this labels are configured, by default:

|               **Label**               |  **Title in Releases**  |
|:-------------------------------------:|:----------------------:|
| `enhancement`, `feature`              | 🚀 Features             |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
| `build`, `ci`, `testing`              | 📦 Build System & CI/CD |
| `breaking`                            | 💥 Breaking Changes     |
| `documentation`                       | 📝 Documentation        |
| `dependencies`                        | ⬆️ Dependencies updates |

You can update it in [`release-drafter.yml`](https://github.com/IslamAlam/pypolsar/blob/master/.github/release-drafter.yml).

GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.

## 🛡 License

[![License](https://img.shields.io/github/license/IslamAlam/pypolsar)](https://github.com/IslamAlam/pypolsar/blob/master/LICENSE)

This project is licensed under the terms of the `GPL-3.0-or-later` license. See [LICENSE](https://github.com/IslamAlam/pypolsar/blob/master/LICENSE) for more details.

## 📃 Citation

```
@misc{pypolsar,
  author = {Earth-Observation},
  title = {PyPolSAR is a python module for Polarimetric Synthetic Aperture Radar (PolSAR) data processing.},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IslamAlam/pypolsar}}
}
```

## Credits

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
