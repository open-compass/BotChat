import math
from tabulate import tabulate
from botchat.utils import *

FONT_FILE = os.environ.get('FONT_FILE', None)

def match_answer(s):
    lines = s.split('\n')
    for _, line in enumerate(lines):
        if line.startswith('Choice:'):
            if 'Conversation 1' in line:
                return 'lose'
            if 'Conversation 2' in line:
                return 'win'
            if 'Both' in line:
                return 'neither'
            if 'Neither' in line:
                return 'both'
    return 'unknown'

def simple_count(data_in):
    data = cp.deepcopy(data_in)
    A, B, ext = data['A'], data['B'], data['extracted']
    res = {}
    for a, b, choice in zip(A, B, ext):
        if a not in res:
            res[a] = defaultdict(lambda: 0)
        if b not in res:
            res[b] = defaultdict(lambda: 0)
        cp_map = {'lose': 'win', 'win': 'lose', 'both': 'both', 'neither': 'neither', 'unknown': 'unknown'}
        res[a][choice] += 1
        res[b][cp_map[choice]] += 1
    return res

def calc_win_rate(data_copy, models):
    data = cp.deepcopy(data_copy)
    
    win = defaultdict(lambda: 0)
    tie = defaultdict(lambda: 0)
    lose = defaultdict(lambda: 0)
    
    for i in range(len(data)):
        v = data.iloc[i]
        o = v['extracted']
        key = v['A'] + ';' + v['B']
        
        if o == 'win':
            win[key] += 1
        elif o == 'lose':
            lose[key] += 1
        elif o in ['both', 'neither']:
            tie[key] += 1
            
    nmodel = len(models)
    cnt = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    ff = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    tot = pd.DataFrame({k: [0] * nmodel for k in models}, index=models)
    for i, k in enumerate(win):
        m1, m2 = k.split(';')
        cnt.at[m1, m2] += win[k]
        cnt.at[m2, m1] += lose[k]
        ff.at[m1, m2] += tie[k]
        ff.at[m2, m1] += tie[k]
        tot.at[m1, m2] += tie[k] + win[k] + lose[k]
        tot.at[m2, m1] += tie[k] + win[k] + lose[k]

    for m1 in models:
        for m2 in models:
            if tot.at[m1, m2]:
                cnt.at[m1, m2] /= tot.at[m1, m2]
                ff.at[m1, m2] /= tot.at[m1, m2]
    return cnt, ff

def analyze(data_file, refm, col_name='gpt4', fout=None, return_table=False):
    # required fields in data:
    # lang, capability, extracted, A, B, index
    if isinstance(data_file, str):
        data = load(data_file)
    else:
        data = data_file
        data_file = 'tmp.tsv'

    nonem = [x != 'EM' for x in data[col_name]]
    double_log(f'{len(data)} comparisons in all, while {sum(nonem)} comparisons are meaningful (two options not exactly the same)', fout)
    data = data[nonem]

    data['extracted'] = [match_answer(ans) for ans in data[col_name]]

    succeed = [not pd.isna(x) for x in data['extracted']]
    succeed_rate = np.mean(succeed)
    double_log(f'{len(succeed)} comparisons in all, succeed to extract {sum(succeed)} answers from judge LLM responses, the successful rate is {succeed_rate * 100:.2f}%', fout)

    data = data[succeed]

    stats = defaultdict(list)

    count_stat = simple_count(data)
    for model in count_stat:
        stat = count_stat[model]
        stats['Model'].append(model)
        winr = stat['win'] / sum(stat.values())
        tier = (stat['both'] + stat['neither']) / sum(stat.values())
        loser = stat['lose'] / sum(stat.values())
        not_bad = (stat['win'] + stat['both']) / sum(stat.values())
        stats['WinRate'].append(f'{winr * 100:.1f}%')
        stats['TieRate'].append(f'{tier * 100:.1f}%')
        stats['LoseRate'].append(f'{loser * 100:.1f}%')
        stats['NotBadRate'].append(f'{not_bad * 100:.1f}%')
        score = (3 * stat['win'] + stat['both'] - stat['neither'] - 3 * stat['lose']) / sum(stat.values())
        stats['Score'].append(score)

    stats = pd.DataFrame(stats)
    stats = stats.sort_values('Score', ascending=False)

    ret_table = {'stats': stats}

    double_log('### Statistics [win / tie / lose / not bad / score (init=0, win +3, both +1, neither -1, lose -3)]', fout)
    double_log('### Score is normalized by the number of comparisons, the normalized range is [-3, 3]', fout)
    double_log(tabulate(stats, headers='keys', tablefmt='pretty'), fout)

    models = list(count_stat.keys())
    models.sort()

    images = []
    wr, dr = calc_win_rate(data, models)

    wr_table = defaultdict(list)
    if refm is not None:
        for m in models:
            if m == refm:
                continue
            wr_table['model'].append(m)
            wr_table['win_rate'].append(wr.at[m, refm])
            wr_table['draw_rate'].append(dr.at[m, refm])
            wr_table['win + draw'].append(dr.at[m, refm] + wr.at[m, refm])
        
        wr_table = pd.DataFrame(wr_table)
        wr_table = wr_table.sort_values('win + draw', ascending=False)

        double_log(f'Win rate compared to {refm}: ', fout)
        double_log(tabulate(wr_table, headers='keys', tablefmt='pretty'), fout)
    
    # im = draw_heatmap(wr, 'Win Rate')
    # images.append(im)
    # im = draw_heatmap(wr + dr, 'Win + Tie Rate')
    # images.append(im)

    ret_table['win'] = wr
    ret_table['wintie'] = wr + dr
    if return_table:
        return ret_table

    # image = stack_image(images, shape=(1, 2))
    # cv2.imwrite('win_rate.png', image)
    dump(data, 'tmp.tsv')
    fout.close()

def chat_analyze():
    parser = argparse.ArgumentParser(
        description="Analyze LLM-based Subjective Evaluation Results. "
    )
    parser.add_argument("data", type=str, help="The LLM Subjective Evaluation Result, in excel format. ")
    parser.add_argument("--col", type=str, default='gpt4', help="The column name. ")
    parser.add_argument("--log", type=str, default='log.txt', help="Log file name. ")
    parser.add_argument("--refm", type=str, default=None, help="Reference Model. ")
    args = parser.parse_args()
    
    analyze(data_file=args.data, refm=args.refm, col_name=args.col, fout=open(args.log, 'w'))
    
if __name__ == '__main__':
    chat_analyze()
