# two_towers
 

To get started with environment:
```
poetry install
```

If poetry is not installed and env consistency is not a priority
```
pip install -r gpu_requirements.txt
```

If do not have `poetry` installed -> https://python-poetry.org/


# Overview of inner workings
Below is an overview of the inner workings of the project. This is not meant to be an exhaustive guide, but rather a guide to help understand the codebase.
## Deployment

Two docker containers are used to run the project:
- 'frontend' - serves Streamlit app
- 'backend' - serves REST API, handles search queries

Use `docker compose up` to start the project.

# Project Roadmap & Todo List

## High Priority
- [ ] Deploy basic server without nginx

## Medium Priority
- [ ] Develop better dataloader for training

## Low Priority
- [ ] Build pipeline for model deployment

## Completed ✅

---
**Legend:**
- [ ] Todo
- [x] Completed
