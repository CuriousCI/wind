import numpy 
import matplotlib.pyplot as plot
import csv
import json

files = ['data/seq.csv', 'data/omp.csv', 'data/mpi.csv', 'data/cuda.csv', 'data/omp_cuda.csv', 'data/mpi_omp.csv']

graphs = [
    'tunnel_size_4',
    # 'tunnel_size_3',
    # 'particles_f_4',
    # 'particles_f_3',
    # 'particles_m_3',
    # 'full_system_3',
    # 'particles_m_4',
    'full_system_4'
]

graph_aggregates = [
    # ['seq', 'omp_1', 'omp_2', 'omp_4', 'omp_8', 'omp_16', 'omp_32'],
    # ['seq', 'cuda'],
    # ['seq', 'mpi_1', 'mpi_2', 'mpi_4', 'mpi_8', 'mpi_16'],
    # ['seq', 'omp_cuda_1', 'omp_cuda_2', 'omp_cuda_4', 'omp_cuda_8', 'omp_cuda_16', 'omp_cuda_32'],
    # ['cuda', 'omp_cuda_1', 'omp_cuda_2', 'omp_cuda_4']
    # ['seq', 'mpi_omp_1_1', 'mpi_omp_1_2', 'mpi_omp_1_4', 'mpi_omp_1_8', 'mpi_omp_2_1', 'mpi_omp_2_2', 'mpi_omp_2_4', 'mpi_omp_2_8', 'mpi_omp_4_1', 'mpi_omp_4_2', 'mpi_omp_4_4', 'mpi_omp_4_8']
    # ['seq', 'mpi_omp_1_1', 'mpi_omp_1_2', 'mpi_omp_1_4', 'mpi_omp_1_8'],
    # ['seq', 'mpi_omp_1_1', 'mpi_omp_2_1', 'mpi_omp_2_2', 'mpi_omp_2_4', 'mpi_omp_2_8'],
    # ['seq', 'mpi_omp_1_1', 'mpi_omp_4_1', 'mpi_omp_4_2', 'mpi_omp_4_4', 'mpi_omp_4_8']
]


data = { graph: {} for graph in graphs }

def stats(data):
    return data, numpy.mean(data), 1.96 * numpy.std(data, ddof=1) / numpy.sqrt(len(data))

for f in files:
    with open(f) as file:
        reader = csv.reader(file)
        for [graph, program, x, time, *rest] in reader:
            try:
                if program not in data[graph]:
                    data[graph][program] = {}

                x = int(x)
                if x not in data[graph][program]:
                    data[graph][program][x] = []

                data[graph][program][x].append(float(time))
            except:
                pass



for graph in data:
    for program in data[graph]:
        for x in data[graph][program]:
            data[graph][program][x] = stats(data[graph][program][x])

# print(json.dumps(data, sort_keys=True, indent=4))

plot.figure(figsize=(6,3))
for graph in data:
    plot.figure(figsize=(5,3))
    plot.figure(figsize=(5,3))
    plot.figure(figsize=(5,3))
    _, ax = plot.subplots()
    for aggregate in graph_aggregates:
        for program in aggregate:
            x_coords = list(data[graph][program].keys())
            x_coords_data = []

            y_coords_data = []
            y_coords_mean = []
            y_coords_conf = []

            for (dat_, mean, conf) in data[graph][program].values(): 
                y_coords_data.append(dat_)
                y_coords_mean.append(mean)
                y_coords_conf.append(conf)

            for (x, y_coords)  in zip(x_coords, y_coords_data):
                for y in y_coords:
                    x_coords_data.append(x)

            x_coords = numpy.array(x_coords)
            # y_coords_data = numpy.array(y_coords_data)
            y_coords_mean = numpy.array(y_coords_mean)
            y_coords_conf = numpy.array(y_coords_conf)

            color = ax._get_lines.get_next_color()

            plot.plot(
                [x_coords, x_coords], 
                [y_coords_mean - y_coords_conf, y_coords_mean + y_coords_conf], 
                '-', linewidth=1.2, color=color
            )
            plot.plot(
                [x_coords - 4, x_coords + 4], 
                [y_coords_mean - y_coords_conf, y_coords_mean - y_coords_conf], 
                '-', linewidth=1.2, color=color
            )
            plot.plot(
                [x_coords - 4, x_coords + 4], 
                [y_coords_mean + y_coords_conf, y_coords_mean + y_coords_conf], 
                '-', linewidth=1.2, color=color
            )

            plot.plot(x_coords, y_coords_mean, label=f'{program} time', linewidth=1.2, color=color)
            # plot.plot(x_coords_data, sum(y_coords_data, []), label=None, marker='o', linestyle='', markersize=2)


        plot.xlabel('size')
        plot.ylabel('time (seconds)')
        plot.title(graph.replace('_3', ' 1000 iter ').replace('_4', ' 10000 iter ').replace('_once', '').replace('_', ' '))
        plot.legend()
        plot.grid(linewidth=0.5, linestyle='--')
        plot.savefig(f'docs/plot/{graph}_{"_".join(aggregate[:3])}')
        plot.gca().set_prop_cycle(None)
        plot.close()

for graph in data:
    for program in data[graph]:
        for x in data[graph][program]:
            data[graph][program][x] = data[graph][program][x][1]

with open('docs/data.csv', 'w') as file:
    graph_aggregates = [
        ['mpi_omp_1_1', 'mpi_omp_2_1', 'mpi_omp_2_2', 'mpi_omp_2_4', 'mpi_omp_2_8', 'mpi_omp_4_1', 'mpi_omp_4_2', 'mpi_omp_4_4', 'mpi_omp_4_8'],
        ['omp_1', 'omp_2', 'omp_4', 'omp_8', 'omp_16', 'omp_32'],
        ['mpi_1', 'mpi_2', 'mpi_4', 'mpi_8', 'mpi_16'],
        ['cuda'],
        ['omp_cuda_1', 'omp_cuda_2', 'omp_cuda_4'],
    ]

    for graph in ['tunnel_size_4', 'full_system_4']:
        for aggregate in graph_aggregates:
            file.write(f'\nspeedup {graph.replace("_3", "_1000_iter_").replace("_4", "_10000_iter_")}_{"_".join(aggregate[0])}\n')
            file.write('program')
            limit = 125
            if graph == 'full_system_4':
                limit = 63
            for x in data[graph]['seq']:
                if x > limit:
                    file.write(',' + str(x))
            file.write('\n')
            for program in aggregate:
                file.write('[' + program)
                for x in data[graph][program]:
                    if x > limit:
                        speedup = data[graph]['seq'][x] / data[graph][program][x]
                        file.write('],[' + str(round(speedup, 2)))
                file.write('],\n')
        file.write('\n\n')

    graph_aggregates = [
        ('mpi_omp_1_1', ['mpi_omp_2_1', 'mpi_omp_2_2', 'mpi_omp_2_4', 'mpi_omp_2_8', 'mpi_omp_1_1', 'mpi_omp_4_1', 'mpi_omp_4_2', 'mpi_omp_4_4', 'mpi_omp_4_8']),
        ('omp_1', ['omp_1', 'omp_2', 'omp_4', 'omp_8', 'omp_16', 'omp_32']),
        ('mpi_1', ['mpi_1', 'mpi_2', 'mpi_4', 'mpi_8', 'mpi_16']),
        ('omp_cuda_1', ['omp_cuda_1', 'omp_cuda_2', 'omp_cuda_4']),
    ]

    for graph in ['tunnel_size_4', 'full_system_4']:
        for (ref, aggregate) in graph_aggregates:
            file.write(f'\nefficiency {graph.replace("_3", "_1000_iter_").replace("_4", "_10000_iter_")}_{"_".join(aggregate[0])}\n')
            file.write('program')
            limit = 125 
            if graph == 'full_system_4':
                limit = 63
            for x in data[graph]['seq']:
                if x > limit:
                    file.write(',' + str(x))
            file.write('\n')
            for program in aggregate:
                file.write('[' + program)
                for x in data[graph][program]:
                    if x > limit:
                        speedup = data[graph]['seq'][x] / data[graph][program][x]
                        p = 1
                        if ref == 'mpi_omp_1_1':
                            p = int(list(program.split('_'))[-1]) * int(list(program.split('_'))[-2])
                        else:
                            p = int(list(program.split('_'))[-1])
                        print(speedup, p, program)
                        efficiency = speedup / p 
                        file.write('],[' + str(round(efficiency, 2)))
                file.write('],\n')
        file.write('\n\n')
