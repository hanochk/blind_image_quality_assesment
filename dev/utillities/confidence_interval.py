# gt - ground truth of objects per blind lpf
# raw estimated lpf
lpf = np.arange(0.01, 1, 0.01)
c = np.cov(raw, gt)
m = (c[0, 1] / c[0, 0])
gt_caret = [m * (x - np.mean(raw)) + np.mean(gt) for x in raw]
dys = [y - x for x, y in zip(gt_caret, gt)]
dys, gt, gt_caret = zip(*[(x, y, z) for x, y, z in sorted(zip(dys, gt, gt_caret), key=lambda x: x[0])])
gt_caret_neg = gt_caret[int(np.round(0.025 * len(dys)))]
gt_caret_pos = gt_caret[int(np.round(0.975 * len(dys)))]
gt_neg = gt[int(np.round(0.025 * len(dys)))]
gt_pos = gt[int(np.round(0.975 * len(dys)))]
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(np.asarray(gt_caret), np.asarray(gt), marker='o', s=5)
g = lambda x, m, epsilon, r: m * x * (1 - np.exp(r * x)) + (m * x - epsilon) * np.exp(r * x)
y = lambda d, x, m, epsilon, r: np.sign(g(x, m, epsilon, r) - x) * np.sqrt(
    abs(g(x, m, epsilon, r) - x) ** 2 + d ** 2) + x
plt.plot(lpf, y(
    d=np.asarray([gamma.interval(alpha=0.975, a=x * N_samples, loc=0, scale=1 / N_samples)[1] - x for x in lpf]),
    x=lpf, m=gt_neg / gt_caret_neg, epsilon=0.02, r=-5))
plt.plot(lpf, y(
    d=np.asarray([x - gamma.interval(alpha=0.975, a=x * N_samples, loc=0, scale=1 / N_samples)[0] for x in lpf]),
    x=lpf, m=gt_pos / gt_caret_pos, epsilon=-0.02, r=-5))