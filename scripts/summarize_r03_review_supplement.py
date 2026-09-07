"""Render existing supplementary evidence, without fitting or selecting models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/r03_review_supplement_2026-09-04'

def main():
    diagnostics=json.loads((ROOT/'reports/references/r03_review_diagnostics_2026-09-04.json').read_text())
    uncertainty=json.loads((ROOT/'reports/references/r03_review_group_uncertainty_2026-09-04.json').read_text())
    replay=json.loads((OUT/'replay_report.json').read_text())
    with (OUT/'predictions.jsonl').open() as f: rows=[json.loads(x) for x in f]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(10,6.5),layout='constrained')
    lines=['# Review 补充实验结果（2026-09-04）','',
        '本文件为冻结结果的事后补充分析，不是新的盲确认，不改变原模型、事件阈值或原始结果。',
        '', '## 1. 重放一致性', '',
        f"仅使用 KBTP 训练集 {replay['training_rows']:,} 个场景拟合原配置，导出 {len(rows):,} 条预测（场景×三个时域）。全部原组级指标的最大绝对差为 {replay['max_abs_difference']}。",
        '历史 locked evidence 汇总文件存在已披露的哈希漂移；它不参与拟合，其他依赖及场景哈希仍严格检查。原失败记录、批准例外及期望/实际哈希保留在重放目录。',
        '', '## 2. 全部组级差值与不确定性', '',
        '差值=B−参照，负值表示B更好。20,000次来源组配对bootstrap、种子170104、点态95%百分位区间。符号检验双侧，排除平局，18项统一Holm校正；最小校正p=0.140625。区间估计平均差值，符号检验评估方向，不能互相替代。', '',
        '|机场|时域/s|比较|平均差值|95%区间|胜/负/平|原始p|Holm p|',
        '|---|---:|---|---:|---|---|---:|---:|']
    for r in uncertainty['records']:
        lo,hi=r['ci95']
        lines.append(f"|{r['airport']}|{r['horizon']}|{r['comparison']}|{r['mean_difference']:.6f}|[{lo:.6f}, {hi:.6f}]|{r['wins']}/{r['losses']}/{r['ties']}|{r['sign_p_two_sided']:.6f}|{r['sign_p_holm_18']:.6f}|")
    lines+=['','## 3. 本地历史参照','','每个KAGC组只使用预测窗已全部结束的更早完整组。第一组预热，不参与任何方法评价；其余9组、每时域1,157场景。历史率、均值、中位数按历史场景汇总估计，评价误差按来源组等权。固定KBTP阈值不重估。这是使用目标机场历史标签的比较，不是纯零样本。','','|指标/方法|30s|120s|300s|','|---|---:|---:|---:|']
    for metric in ('count','probability'):
        results=diagnostics['local_chronological_KAGC']
        for method in results['30'][metric]:
            values=[results[str(h)][metric][method]['equal_group_mean'] for h in (30,120,300)]
            lines.append('|'+metric+'/'+method+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    lines+=['','30s历史中位数始终为0，其MAE优于B；不能声称B优于所有朴素计数方法。B的概率分数在该共同子集上优于两种事件率参照。该子集的300s结果不能替换原十组迁移结论。', '', '## 4. 报告陈旧度分层', '',
        'fresh=场景当前报告最大年龄≤10s，older=>10s。阳性场景数是超过固定计数阈值的场景数，不是事件总次数。分层构成及事件率不同，不作因果解释。', '',
        '|机场|时域|层|组数|场景数|阳性场景数|事件次数和|B MAE|A MAE|B Brier|事件率Brier|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ai,a in enumerate(('KBTP','KAGC')):
        for hi,h in enumerate((30,120,300)):
            d=diagnostics['by_airport'][a][str(h)]
            bins=d['reliability_pooled']
            valid=[b for b in bins if b['n']]
            ax=axes[ai,hi]
            ax.plot([0,1],[0,1],color='#bbbbbb',linestyle='--',linewidth=1)
            ax.plot([b['mean_probability'] for b in valid],[b['observed_rate'] for b in valid],color='#477eaa',linewidth=1)
            ax.scatter([b['mean_probability'] for b in valid],[b['observed_rate'] for b in valid],s=[20+ min(b['n'],600)/7 for b in valid],color='#8fbbd9',edgecolors='#477eaa')
            for b in valid:
                ax.annotate(str(b['n']),(b['mean_probability'],b['observed_rate']),xytext=(3,4),textcoords='offset points',fontsize=6)
            ax.set(xlim=(-.02,1.02),ylim=(-.02,1.04),title=f'{a} | {h} s',xlabel='Mean predicted probability',ylabel='Observed event fraction')
            ax.spines[['top','right']].set_visible(False)
            for name,s in d['freshness'].items():
                selected=[r for r in rows if r['airport']==a and r['horizon']==h and (r['max_report_age_s']<=10)==(name=='fresh')]
                b=s['count_B']
                values=[s[k]['equal_group_mean'] for k in ('count_B','count_A','prob_B','prevalence')]
                lines.append(f"|{a}|{h}|{name}|{b['n_groups']}|{b['n_scenes']}|{b['n_positive']}|{sum(r['y'] for r in selected):.0f}|"+'|'.join(f'{v:.6f}' for v in values)+'|')
    fig.suptitle('Reliability by airport and horizon (pooled scenes; labels = bin counts)',fontsize=11)
    fig.savefig(OUT/'reliability.png',dpi=180,facecolor='white')
    plt.close(fig)
    lines+=['','## 5. 可靠性图与分箱数据','','分箱固定为10个等宽概率区间，p=1归入最后一箱；空箱不画点。曲线为合并场景描述，不代表来源组独立置信区间。数字为该箱场景数，稀疏高概率箱不可过度解释。','',f'![可靠性图]({(OUT/"reliability.png").as_posix()})','', '|机场|时域|箱区间|场景数|来源组数|平均概率|阳性比例|','|---|---:|---|---:|---:|---:|---:|']
    for a in ('KBTP','KAGC'):
        for h in (30,120,300):
            for b in diagnostics['by_airport'][a][str(h)]['reliability_pooled']:
                v='—|—' if not b['n'] else f"{b['mean_probability']:.6f}|{b['observed_rate']:.6f}"
                lines.append(f"|{a}|{h}|{b['lower']:.1f}–{b['upper']:.1f}|{b['n']}|{b['n_groups']}|{v}|")
    lines+=['','## 6. 当前链路耗时','','31个确定性抽样场景、3–11架飞机，每场景预热一次再测一次。六个输出=三个时域的计数和概率。现有特征函数内部仍计算C/CPA。', '', '|阶段|P50/ms|P95/ms|','|---|---:|---:|']
    for k,v in replay['latency']['summary']['all'].items():
        if isinstance(v,dict):lines.append(f"|{k}|{v['p50']:.4f}|{v['p95']:.4f}|")
    cold_path=OUT/'cold_start_report.json'
    if cold_path.exists():
        cold=json.loads(cold_path.read_text())
        c=cold['summary']
        lines+=['',f"独立进程冷启动补测：{cold['n']}次新Python进程，同一个固定KBTP场景，不训练。进程启动到退出（包括导入、模型和单场景读取、六输出）P50/P95={c['process_wall_ms']['p50']:.2f}/{c['process_wall_ms']['p95']:.2f}ms；其中导入P50={c['import_ms']['p50']:.2f}ms，首次纯计算P50={c['first_compute_ms']['p50']:.2f}ms。",'操作系统磁盘缓存未清除，不等于整机或磁盘冷启动；5次样本的P95仅为描述性分位数。常驻服务通常只支付一次初始化成本，不能把每次预测都算作进程启动。']
    env=replay['environment']
    lines+=['',f"环境：{env['platform']}；{env['processor']}；{env['logical_cpu_count']}逻辑CPU；scikit-learn {env['sklearn']}、NumPy {env['numpy']}。",
        '以上是内存历史缓冲到输出的计算，不包括网络、磁盘、缓冲维护或初始两分钟历史积累。逐场景首调用不是进程冷启动；冷启动单独记录，不与以上暖调用混称。', '', '## 7. 结论与尚未覆盖的范围', '',
        '可支持的贡献：可复现的新生事件目标、成对信息的受控增量评估、低计算成本的条件概率筛查。不能支持：全面优于朴素基线、校正后统计显著、对陈旧度稳健、实际运行风险认证或已完成端到端部署。',
        '本轮不新增意图感知CPA、非高峰数据、第三机场或深模型；这些仍是计划中明确分离的后续工作。先冻结本补充实验，再改稿并严格采用用户提供的IEEE模板。']
    (ROOT/'reports/R03_REVIEW_SUPPLEMENT_RESULTS_2026-09-04.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Rendered reliability plot and complete supplementary result report.')

if __name__=='__main__': main()
