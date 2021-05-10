###############################
#
# Created by Patrik Valkovic
# 5/5/2021
#
###############################
import wandb
api = wandb.Api()
PER_PAGE = 2000

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    'config.device.value': 'cpu',
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'CPU runs: {len(runs)} running for {running_time}')

api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    'config.device.value': 'cuda',
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'CUDA runs: {len(runs)} running for {running_time}')

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$and': [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': 'pso'},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'PSO: {len(runs)} running for {running_time}')

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$and': [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.run_type.value': 'test'},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'Test runs: {len(runs)} running for {running_time}')

api = wandb.Api()
runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$or': [
        {'state': 'crashed'},
        {'config.run_failed.value': True},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'Failed runs: {len(runs)} running for {running_time}')

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$and': [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.run_type.value': 'hyperparametersearch'},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'Hyperparameter search: {len(runs)} running for {running_time}')

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$and': [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': {'$in': [
            'ga_1',
            'ga_scaling',
            'ga_selection',
            'ga_varcount',
            'ga_c',
            'ga_clausecount',
            'ga',
        ]}},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'GA: {len(runs)} running for {running_time}')

runs = api.runs(f'kowalsky/thesis', per_page=PER_PAGE, filters={
    '$and': [
        {'state': 'finished'},
        {'config.run_failed.value': False},
        {'config.alg_group.value': {'$in': [
            'es_elitism',
            'es_schema',
            'es_mutation',
            'es_crossover',
        ]}},
    ]
})
running_time = sum([run.summary['_runtime'] for run in filter(lambda r: '_runtime' in r.summary, runs)])
print(f'ES: {len(runs)} running for {running_time}')
