import numpy as np
from numpy.fft import fft, fftshift
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt


def spectralOmni(I, window=None):
    old_imp = False
    if old_imp:
        I = I - np.mean(I.ravel())
    # fft_size = np.power(2,(nextpow2(max(np.size(I)))) * 2
    fft_size = 2*np.power(2,(1+int(np.log2(max(np.shape(I))))))
    # fft_size = np.power(2,(int(np.log2(max(np.shape(I))))))
    if window is not None:   #https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic1.htm#window
        han = np.hanning(I.shape[0])[:, None]
        han2d = np.sqrt(np.dot(han, han.T))
        I = I*han2d
    if old_imp:
        from scipy.fftpack import fft2  # old =>new from numpy.fft import fft, fftshift
        spec = fft2(I, shape=(fft_size, fft_size))/fft_size
    else:
        spec = np.fft.fft2(I, s=(fft_size, fft_size))/fft_size

    angles = np.arange(-np.pi/2,np.pi/2 - np.pi/fft_size,np.pi/fft_size)#(-np.pi/2: np.pi / fft_size: np.pi/2 - np.pi / fft_size)  #N*1
    rad = np.arange(0, fft_size/2)#(0:fft_size / 2-1)                               # 1*N
    x = np.round(np.cos(angles)[:, np.newaxis]*rad[np.newaxis,:])                           # N*N
    y = np.round(np.sin(angles)[:, np.newaxis]*rad[np.newaxis,:])                           # N*N
    y[y < 0] = y[y < 0] + fft_size
    k = (x * fft_size + y).astype('int')                       # 1 for matlab
    spec1d = np.mean(np.power(np.abs(spec.ravel()[k.ravel()]).reshape(k.shape), 2), axis=0)  #spec = mean(abs(spec(k)).^2);
    # plt.imshow(fftpack.fftshift(np.abs(spec)))
    # plt.semilogy(10 * np.arange(0, 256) / 256, g)
    return spec1d, fftpack.fftshift(spec)


def hanning(xsize, ysize, invert=False):
    """
    Make a 2D hanning kernel from the seperable 1D hanning kernels. The
    default kernel peaks at 1 at the center and falls to zero at the edges.
    Use invert=True to make an inner mask that is 0 at the center and rises
    to 1 at the edges.

    """
    mask1D_x = np.hanning(xsize)
    mask1D_y = np.hanning(ysize)
    mask = np.outer(mask1D_x, mask1D_y)
    if invert:
        mask = -1.0 * mask + 1.0

    return mask


import numpy as np


def azimuthalAverage(image, center=None, ignoreNAN=False):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    Output:
    radii, radial_prof, stddev_prof, nr
    radii - array of radii
    radial_prof - array of average values for each radius bin
    stddev_prof - array of stddev value for each radius bin
    nr - the number of pixels that went into each radius bin
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius

    radii = np.arange(len(rind) - 1, dtype=float) + 1
    radii *= deltar[rind[:-1]]
    nr = rind[1:] - rind[:-1]  # number of pixels in radius bin
    radial_prof = np.zeros(len(nr), dtype=float)
    stddev_prof = np.zeros(len(nr), dtype=float)

    for rr in range(len(rind) - 1):
        if not ignoreNAN:
            radial_prof[rr] = i_sorted[rind[rr]:rind[rr + 1]].mean()
            stddev_prof[rr] = i_sorted[rind[rr]:rind[rr + 1]].std()
        else:
            all_val = i_sorted[rind[rr]:rind[rr + 1]]
            good = np.isfinite(all_val)

            radial_prof[rr] = all_val[good].mean()
            stddev_prof[rr] = all_val[good].std()
            nr[rr] = len(good)

    return radii, radial_prof, stddev_prof, nr

def mtf(image):
    # https://github.com/jluastro/JLU-python-code/blob/master/jlu/nirc2/mtf/mtf.py
    # Apodize the image with a Hanning kernal to enforce periodicity
    szx = image.shape[0]
    szy = image.shape[1]
    han = hanning(szx, szy)
    # img_skysub = image - skyMode

    fftim = fftpack.fft2(image * han) / (szx * szy)
    absim = np.real(fftim * fftim.conjugate())
    absim[0, 0] = np.nan  # don't count the DC component
    wrapim = fftpack.fftshift(absim)  # this is the 2D power spectrum
    ind = np.where(np.isfinite(wrapim) == False)
    xcen = ind[0][0]
    ycen = ind[1][0]
    # HK: it takes only the radial spectrum i.e from image of [256x256] = >[256/sqrt(2)x256/sqrt(2)]
    tmp = azimuthalAverage(wrapim, center=[xcen, ycen],
                                         ignoreNAN=True)
    # tmp = radialProfile.azimuthalAverage(wrapim, center=[xcen, ycen],
    #                                      ignoreNAN=True)
    pix = tmp[0]
    value = tmp[1]
    rms = tmp[2]
    npts = tmp[3]
    error = rms / np.sqrt(npts)
    return value, error, npts

def plot_2d_stat(spectOmni_acm, fname='spectral_sum'):
    sum_spect = spectOmni_acm.mean(axis=0)

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig, ax = plt.subplots()
    ax.semilogy(sum_spect+np.finfo(float).eps)
    ax.set_ylim(0.0000001, sum_spect.max())
    ax.set_title("Spectrum sum N={}".format(spectOmni_acm.shape[0]))
    ax.set_ylabel('mean')
    ax.set_xlabel('Spectral bins')
    ax.grid()
    plt.show()
    fig.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')


def plot_3d_stat(spectOmni_acm, fname='hist_res'):
    hist_acm = []
    for dim in np.arange(spectOmni_acm.shape[1]):
        # a_e = np.histogram(spectOmni_acm[:, dim]+np.finfo(float).eps, bins=min(100, spectOmni_acm.shape[0]))
        a_e = np.histogram(spectOmni_acm[:, dim] + np.finfo(float).eps, bins=min(100, spectOmni_acm.shape[0]),
                           range=(0, spectOmni_acm.max()))
        hist_acm += [a_e[0]]
    bins_loc_e = (a_e[1][0:-1] + a_e[1][1:]) / 2
    hist_spect = np.concatenate(hist_acm, axis=0).reshape(spectOmni_acm.shape[1], -1)
    # plt.imshow(hist_spect)
    plt.imshow(np.log2(hist_spect), cmap='jet')
    plt.title('Spectrum log2 histogram ')

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # x = np.linspace(0, min(100, spectOmni_acm.shape[0]), min(100, spectOmni_acm.shape[0]))
    y = np.linspace(0, 512, 512)
    X, Y = np.meshgrid(bins_loc_e, y)
    surf = ax.plot_surface(X, Y, hist_spect, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('histogram')
    ax.set_ylabel('Spectral bins')
    fig.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')
    print(ax.elev)
    print(ax.azim)
    print(ax.dist)
    plt.show()
    ax.view_init(elev=83.56, azim=-79)
    plt.show()
    fig.savefig(fname + 'BEV.pdf')

