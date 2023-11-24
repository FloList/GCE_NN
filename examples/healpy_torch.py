import torch

jrll = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64)
jpll = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)


# pix2ang
def pix2ang(ipix, nside):
    # NOTE: This assumes RING ordering!
    if nside < 1 or nside > 2**30:
        raise ValueError("Nside should be power of 2 >0 and < 2^30")

    nsidesq = nside * nside
    npix = 12 * nsidesq  # total number of pixels

    if (ipix < 0).any() or (ipix > npix - 1).any():
        raise ValueError("ipix out of range calculated from nside")

    ipix1 = ipix + 1  # in [1, npix]
    nl2 = 2 * nside
    nl4 = 4 * nside
    ncap = 2 * nside * (nside - 1)  # points in each polar cap, =0 for nside =1

    # North polar cap
    condition_north_polar_cap = ipix1 <= ncap
    hip_north = ipix1[condition_north_polar_cap] / 2.0
    fihip_north = torch.floor(hip_north)  # get integer part of hip
    iring_north = torch.floor(torch.sqrt(hip_north - torch.sqrt(fihip_north))) + 1  # counted from north pole
    iphi_north = ipix1[condition_north_polar_cap] - 2 * iring_north * (iring_north - 1)
    theta_north = torch.acos(1.0 - iring_north * iring_north / (3.0 * nsidesq))
    phi_north = ((iphi_north - 0.5) * torch.pi / (2.0 * iring_north))

    # Equatorial region
    condition_equatorial = (ipix1 > ncap) & (ipix1 <= nl2 * (5 * nside + 1))
    ip_equatorial = ipix1[condition_equatorial] - ncap - 1
    iring_equatorial = (ip_equatorial // nl4) + nside  # counted from North pole
    iphi_equatorial = ip_equatorial % nl4 + 1
    fodd_equatorial = 0.5 * (1. + ((iring_equatorial + nside) % 2))  # 1 if iring+nside is odd, 1/2 otherwise
    theta_equatorial = torch.acos((nl2 - iring_equatorial) / (1.5 * nside))
    phi_equatorial = ((iphi_equatorial - fodd_equatorial) * torch.pi / (2.0 * nside))

    # South polar cap
    condition_south_polar_cap = ipix1 > nl2 * (5 * nside + 1)
    ip_south = npix - ipix1[condition_south_polar_cap] + 1
    hip_south = ip_south / 2.0
    fihip_south = torch.floor(hip_south)
    iring_south = torch.floor(torch.sqrt(hip_south - torch.sqrt(fihip_south))) + 1  # counted from South pole
    iphi_south = 4 * iring_south + 1 - (ip_south - 2 * iring_south * (iring_south - 1))
    theta_south = torch.acos(-1.0 + iring_south * iring_south / (3.0 * nsidesq))
    phi_south = ((iphi_south - 0.5) * torch.pi / (2.0 * iring_south))

    # Combine results
    theta = torch.zeros_like(ipix, dtype=torch.get_default_dtype())
    phi = torch.zeros_like(ipix, dtype=torch.get_default_dtype())

    theta[condition_north_polar_cap] = theta_north
    phi[condition_north_polar_cap] = phi_north

    theta[condition_equatorial] = theta_equatorial
    phi[condition_equatorial] = phi_equatorial

    theta[condition_south_polar_cap] = theta_south
    phi[condition_south_polar_cap] = phi_south

    return torch.stack([theta, phi], dim=1)


# Bit compression
def compress_bits(v):
    res = torch.bitwise_and(v,
                            torch.tensor(0x5555555555555555, dtype=torch.int64))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 1)),
                            torch.tensor(0x3333333333333333, dtype=torch.int64))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 2)),
                            torch.tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 4)),
                            torch.tensor(0x00FF00FF00FF00FF, dtype=torch.int64))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 8)),
                            torch.tensor(0x0000FFFF0000FFFF, dtype=torch.int64))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 16)),
                            torch.tensor(0x00000000FFFFFFFF, dtype=torch.int64))
    return res


# T_HPD class
# A structure describing the discrete Healpix coordinate system.
# f takes values in [0, 11], x and y lie in [0, nside).
class HPD:
    def __init__(self, x, y, f):
        self.x = x
        self.y = y
        self.f = f


# Convert from nested to HPD
def nest2hpd(nside, pix):
    npface_ = nside * nside
    p2 = pix & (npface_ - 1)
    return HPD(compress_bits(p2), compress_bits(p2 >> 1), pix // npface_)


# Convert from HPD to ring
def hpd2ring(nside, hpd):
    nl4 = 4 * nside
    jr = (jrll[hpd.f] * nside) - hpd.x - hpd.y - 1

    condition1 = jr < nside
    condition2 = jr > 3 * nside

    jp_condition1 = (jpll[hpd.f] * jr + hpd.x - hpd.y + 1) // 2
    jp_condition2 = (jpll[hpd.f] * (nl4 - jr) + hpd.x - hpd.y + 1) // 2
    jp_condition3 = (jpll[hpd.f] * nside + hpd.x - hpd.y + 1 + ((jr - nside) & 1)) // 2

    jp_condition1 = torch.where(jp_condition1 > nl4, jp_condition1 - nl4, torch.where(jp_condition1 < 1, jp_condition1 + nl4, jp_condition1))
    jp_condition2 = torch.where(jp_condition2 > nl4, jp_condition2 - nl4, torch.where(jp_condition2 < 1, jp_condition2 + nl4, jp_condition2))
    jp_condition3 = torch.where(jp_condition3 > nl4, jp_condition3 - nl4, torch.where(jp_condition3 < 1, jp_condition3 + nl4, jp_condition3))

    result_condition1 = 2 * jr * (jr - 1) + jp_condition1 - 1
    result_condition2 = 12 * nside * nside - 2 * (nl4 - jr + 1) * (nl4 - jr) + jp_condition2 - 1
    result_condition3 = 2 * nside * (nside - 1) + (jr - nside) * nl4 + jp_condition3 - 1

    return torch.where(condition1, result_condition1, torch.where(condition2, result_condition2, result_condition3))



# Convert from nested to ring
def nest2ring(nside, ipnest):
    return hpd2ring(nside, nest2hpd(nside, ipnest))


if __name__ == "__main__":
    # TESTING
    import healpy as hp
    import numpy as np

    torch.set_default_dtype(torch.float64)
    nside_testing = 256

    # Comparison of nside2npix
    out_torch = pix2ang(torch.arange(hp.nside2npix(nside_testing)), nside_testing)
    out_hp = hp.pix2ang(nside_testing, np.arange(hp.nside2npix(nside_testing)), nest=False, lonlat=False)
    print("Max. difference in theta: ", torch.max(torch.abs(out_torch[:, 0] - out_hp[0])))
    print("Max. difference in phi: ", torch.max(torch.abs(out_torch[:, 1] - out_hp[1])))
    print(np.asarray(out_torch[:10, 0]), out_hp[0][:10], sep="\n")

    # Comparison of nest2ring
    out_torch = nest2ring(nside_testing, torch.arange(hp.nside2npix(nside_testing)))
    out_hp = hp.nest2ring(nside_testing, np.arange(hp.nside2npix(nside_testing)))
    assert torch.all(out_torch == torch.from_numpy(out_hp)), "nest2ring does not agree with healpy.nest2ring!"

