import numpy as np
import matplotlib.pyplot as plot
import csv
import json

graphs = [
    'tunnel_size_4',
    'tunnel_size_3',
    'particles_f_4',
    'particles_f_3',
    'particles_m_3',
    'full_system_3',
    'particles_m_4',
    'full_system_4_once'
]

data = { graph: {} for graph in graphs }
files = ['data/logs_seq_.csv', 'data/logs_cuda.csv', 'data/logs_omp_.csv']

def stats(data):
    return np.mean(data)

for f in files:
    with open(f) as file:
        reader = csv.reader(file)
        for [graph, prog, size, time, *rest] in reader:
            try:
                if prog not in data[graph]:
                    data[graph][prog] = {}

                size = int(size)

                if size not in data[graph][prog]:
                    data[graph][prog][size] = []

                data[graph][prog][size] += [float(time)]
            except:
                pass



for graph in data:
    for prog in data[graph]:
        for size in data[graph][prog]:
            data[graph][prog][size] = stats(data[graph][prog][size])

print(data)

with open('docs/data.csv', 'w') as file:
    for graph in data:
        file.write(graph.replace('_3', ' 1000 iter ').replace('_4', ' 10000 iter ').replace('_once', '') + '\n')
        file.write('prog')
        for size in data[graph]['seq_']:
            file.write(',' + str(size))
        file.write('\n')
        for prog in data[graph]:
            file.write('wind_' + prog)
            for size in data[graph][prog]:
                file.write(',' + str(round(data[graph][prog][size], 4)))
            file.write('\n')
        file.write('\n')



for graph in data:
    plot.figure(figsize=(10,6))
    for prog in data[graph]:
        if prog == 'cuda':
            continue

        x_coords = data[graph][prog].keys()
        y_coords = data[graph][prog].values()

        y_coords = [data[graph]['seq_'][x] / y for (x, y) in zip(x_coords, y_coords)]

        # plot.plot(x_coords, y_coords, label=f'{prog} speedup')
        if prog != 'seq_':
            y_coords = [data[graph][prog[:3] + '_1'][x] / y for (x, y) in zip(x_coords, y_coords)]
            plot.plot(x_coords, y_coords, label=f'{prog} efficiency')
        else:
            pass

        plot.savefig(f'docs/{graph}')


    plot.xlabel('size')
    plot.ylabel('time')
    plot.title(graph.replace('_3', ' 1000 iter ').replace('_4', ' 10000 iter ').replace('_once', ''))
    plot.legend()
    plot.grid()
    plot.savefig(f'docs/{graph}')
    plot.close()


# print(data[graph][prog].items())
# x_coords, y_coords = data[graph][prog].items()
# y_coords = [mean for (mean, _) in y_coords]
# x_coords = []
# y_coords = []
# for (x, y) in data[graph][prog]:
#     x_coords += [x]
#     y_coords += [y]
# plot.errorbar(positions, means, yerr=cis, fmt='o', color='blue', 
#              capsize=5, capthick=2, label='Mean ± 95% CI')

    # mean = []
    # ci = []

    # for group in measurements:
    #     mean = np.mean(group)
    #     ci = 1.96 * np.std(group, ddof=1) / np.sqrt(len(group))
    #     means.append(mean)
    #     cis.append(ci)
    #
    # return means, cis



# data_seq_ = 9
# Sample data (feel free to modify these values)
# measurements = [
#     [98, 102, 101, 97, 103],  # Group 1
#     [150, 148, 152, 149, 151], # Group 2
#     [75, 76, 74, 77, 75]       # Group 3
# ]
#
#
# # Calculate statistics
# means, cis = calculate_stats(measurements)
#
# # Create positions for groups
# positions = np.arange(len(measurements))
#
# # Create figure and axis
# plot.figure(figsize=(10, 6))
#
# # Plot individual measurements
# for i, group in enumerate(measurements):
#     # Random jitter for better visibility
#     jitter = np.random.uniform(-0.1, 0.1, len(group))
#     plot.scatter(positions[i] + jitter, group, 
#                color='gray', alpha=0.5, label='Individual Measurements' if i == 0 else "")
#
# # Plot means and confidence intervals
# plot.errorbar(positions, means, yerr=cis, fmt='o', color='blue', 
#              capsize=5, capthick=2, label='Mean ± 95% CI')
#
# # Customize the plot
# # plot.xticks(positions, [f'Group {i+1}' for i in range(len(me
# import matplotlib.pyplot as plot
#
# # Create figure with specific dimensions
# fig, ax = plot.subplots(figsize=(8, 6))
#
# # Plot data
# ax.plot([1, 2, 3], [1, 4, 9])
#
# # Customize plot
# ax.set_title('Example Plot')
# ax.grid(True)
#
# # Save figure with DPI control
# fig.savefig('docs/plot.png', dpi=300, bbox_inches='tight')
