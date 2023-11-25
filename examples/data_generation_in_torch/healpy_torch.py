import torch

# pix2ang
def pix2ang(ipix, nside):
    # NOTE: This assumes RING ordering!
    if nside < 1:
        raise ValueError("Nside should be power of 2 >0")

    nsidesq = nside ** 2
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
    theta = torch.zeros_like(ipix, dtype=torch.get_default_dtype(), device=ipix.device)
    phi = torch.zeros_like(ipix, dtype=torch.get_default_dtype(), device=ipix.device)

    theta[condition_north_polar_cap] = theta_north
    phi[condition_north_polar_cap] = phi_north

    theta[condition_equatorial] = theta_equatorial
    phi[condition_equatorial] = phi_equatorial

    theta[condition_south_polar_cap] = theta_south
    phi[condition_south_polar_cap] = phi_south

    return torch.stack([theta, phi], dim=1)


# ang2pix
def zphi2pix(nside, z, phi):
    ipix1 = torch.zeros_like(z, dtype=torch.int64, device=z.device)
    tt = phi / (0.5 * torch.pi)  # in [0, 4]
    za = torch.abs(z)
    nl2 = 2 * nside
    nl4 = 4 * nside
    ncap = nl2 * (nside - 1)  # number of pixels in the north polar cap
    npix = 12 * nside ** 2

    condition_equatorial = za < 2.0 / 3.0

    jp_equatorial = (nside * (0.5 + tt - 0.75 * z)).long()
    jm_equatorial = (nside * (0.5 + tt + 0.75 * z)).long()

    ir_equatorial = nside + 1 + jp_equatorial - jm_equatorial

    kshift_equatorial = torch.where(ir_equatorial % 2 == 0,
                                    torch.tensor(1, device=z.device),
                                    torch.tensor(0, device=z.device))

    ip_equatorial = jp_equatorial + jm_equatorial - nside + kshift_equatorial + 1 + 2 * nl4
    ip_equatorial = torch.bitwise_and(torch.bitwise_right_shift(ip_equatorial, 1), (nl4 - 1)) + 1

    ipix1 = torch.where(condition_equatorial, ncap + nl4 * (ir_equatorial - 1) + ip_equatorial, ipix1)

    tp_polar = tt - torch.floor(tt)
    tmp_polar = torch.sqrt(3.0 * (1.0 - za))
    jp_polar = (nside * tp_polar * tmp_polar).long()
    jm_polar = (nside * (1.0 - tp_polar) * tmp_polar).long()

    ir_polar = jp_polar + jm_polar + 1
    ip_polar = (tt * ir_polar + 1).long()
    ip_polar = torch.where(ip_polar > 4 * ir_polar, ip_polar - 4 * ir_polar, ip_polar)

    ipix1_polar = 2 * ir_polar * (ir_polar - 1) + ip_polar
    ipix1_polar = torch.where(z <= 0.0, npix - 2 * ir_polar * (ir_polar + 1) + ip_polar, ipix1_polar)

    ipix1 = torch.where(~condition_equatorial, ipix1_polar, ipix1)

    return ipix1 - 1  # in [0, npix-1]


def ang2pix(nside, theta, phi):
    if nside < 1:
        raise ValueError("Nside should be a power of 2 > 0")
    if torch.any(theta < 0.0) or torch.any(theta > torch.pi):
        raise ValueError("Theta out of range [0, pi]")

    z = torch.cos(theta)

    phi = torch.where(phi >= 2.0 * torch.pi, phi - 2.0 * torch.pi, phi)
    phi = torch.where(phi < 0.0, phi + 2.0 * torch.pi, phi)

    return zphi2pix(nside, z, phi)


# Bit compression
def compress_bits(v):
    res = torch.bitwise_and(v,
                            torch.tensor(0x5555555555555555, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 1)),
                            torch.tensor(0x3333333333333333, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 2)),
                            torch.tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 4)),
                            torch.tensor(0x00FF00FF00FF00FF, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 8)),
                            torch.tensor(0x0000FFFF0000FFFF, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_right_shift(res, 16)),
                            torch.tensor(0x00000000FFFFFFFF, dtype=torch.int64, device=v.device))
    return res


# Bit spreading
def spread_bits(v):
    res = torch.bitwise_and(v,
                            torch.tensor(0xFFFFFFFF, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_left_shift(res, 16)),
                            torch.tensor(0x0000FFFF0000FFFF, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_left_shift(res, 8)),
                            torch.tensor(0x00FF00FF00FF00FF, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_left_shift(res, 4)),
                            torch.tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_left_shift(res, 2)),
                            torch.tensor(0x3333333333333333, dtype=torch.int64, device=v.device))
    res = torch.bitwise_and(torch.bitwise_xor(res, torch.bitwise_left_shift(res, 1)),
                            torch.tensor(0x5555555555555555, dtype=torch.int64, device=v.device))
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
    p2 = torch.bitwise_and(pix, (npface_ - 1))
    return HPD(compress_bits(p2), compress_bits(torch.bitwise_right_shift(p2, 1)), pix // npface_)


# Convert from HPD to ring
def hpd2ring(nside, hpd):
    jrll = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64, device=hpd.x.device)
    jpll = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64, device=hpd.x.device)

    nl4 = 4 * nside
    jr = (jrll[hpd.f] * nside) - hpd.x - hpd.y - 1

    condition1 = jr < nside
    condition2 = jr > 3 * nside

    jp_condition1 = (jpll[hpd.f] * jr + hpd.x - hpd.y + 1) // 2
    jp_condition2 = (jpll[hpd.f] * (nl4 - jr) + hpd.x - hpd.y + 1) // 2
    jp_condition3 = (jpll[hpd.f] * nside + hpd.x - hpd.y + 1 + ((jr - nside) & 1)) // 2

    jp_condition1 = torch.where(jp_condition1 > nl4, jp_condition1 - nl4,
                                torch.where(jp_condition1 < 1, jp_condition1 + nl4, jp_condition1))
    jp_condition2 = torch.where(jp_condition2 > nl4, jp_condition2 - nl4,
                                torch.where(jp_condition2 < 1, jp_condition2 + nl4, jp_condition2))
    jp_condition3 = torch.where(jp_condition3 > nl4, jp_condition3 - nl4,
                                torch.where(jp_condition3 < 1, jp_condition3 + nl4, jp_condition3))

    result_condition1 = 2 * jr * (jr - 1) + jp_condition1 - 1
    result_condition2 = 12 * nside * nside - 2 * (nl4 - jr + 1) * (nl4 - jr) + jp_condition2 - 1
    result_condition3 = 2 * nside * (nside - 1) + (jr - nside) * nl4 + jp_condition3 - 1

    return torch.where(condition1, result_condition1, torch.where(condition2, result_condition2, result_condition3))


# Convert from nested to ring
def nest2ring(nside, ipnest):
    return hpd2ring(nside, nest2hpd(nside, ipnest))


# Convert from HPD to nested
def hpd2nest(nside, hpd):
    return (hpd.f * nside ** 2) + spread_bits(hpd.x) + (torch.bitwise_left_shift(spread_bits(hpd.y), 1))


# Convert from ring to HPD
def isqrt(x):
    return x.long().sqrt().long()


def ring2hpd(nside, pix):
    jrll = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64, device=pix.device)
    jpll = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64, device=pix.device)

    ncap_ = 2 * nside * (nside - 1)
    npix_ = 12 * nside ** 2

    mask_north = pix < ncap_
    mask_equatorial = (pix >= ncap_) & (pix < (npix_ - ncap_))

    iring_north = torch.where(mask_north, torch.bitwise_right_shift(1 + isqrt(1 + 2 * pix), 1),
                              torch.ones_like(pix)).long()
    iphi_north = torch.where(mask_north, pix + 1 - 2 * iring_north * (iring_north - 1), torch.zeros_like(pix)).long()
    face_north = torch.where(mask_north, torch.floor_divide(iphi_north - 1, iring_north), torch.zeros_like(pix)).long()
    irt_north = iring_north - (jrll[face_north] * nside) + 1
    ipt_north = 2 * iphi_north - jpll[face_north] * iring_north - 1
    ipt_north = torch.where(ipt_north >= 2 * nside, ipt_north - 8 * nside, ipt_north)

    iring_equatorial = torch.where(mask_equatorial, torch.floor_divide(pix - ncap_, 4 * nside) + nside,
                                   torch.zeros_like(pix)).long()
    iphi_equatorial = torch.where(mask_equatorial, torch.remainder(pix - ncap_, 4 * nside) + 1,
                                  torch.zeros_like(pix)).long()
    kshift_equatorial = torch.where(mask_equatorial, torch.bitwise_and(iring_equatorial + nside, 1),
                                    torch.zeros_like(pix)).long()
    ire_equatorial = iring_equatorial - nside + 1
    irm_equatorial = 2 * nside + 2 - ire_equatorial
    ifm_equatorial = torch.floor_divide(iphi_equatorial - ire_equatorial // 2 + nside - 1, nside).long()
    ifp_equatorial = torch.floor_divide(iphi_equatorial - irm_equatorial // 2 + nside - 1, nside).long()
    face_equatorial = torch.where(ifp_equatorial == ifm_equatorial, ifp_equatorial | 4,
                                  torch.where(ifp_equatorial < ifm_equatorial, ifp_equatorial,
                                              ifm_equatorial + 8)).long()
    irt_equatorial = iring_equatorial - (jrll[face_equatorial] * nside) + 1
    ipt_equatorial = 2 * iphi_equatorial - jpll[face_equatorial] * nside - kshift_equatorial - 1
    ipt_equatorial = torch.where(ipt_equatorial >= 2 * nside, ipt_equatorial - 8 * nside, ipt_equatorial)

    iring_south = torch.where(~mask_north & ~mask_equatorial,
                              torch.bitwise_right_shift(1 + isqrt(2 * (npix_ - pix) - 1), 1),
                              torch.ones_like(pix)).long()
    iphi_south = torch.where(~mask_north & ~mask_equatorial,
                             4 * iring_south + 1 - ((npix_ - pix) - 2 * iring_south * (iring_south - 1)),
                             torch.zeros_like(pix)).long()
    face_south = torch.where(~mask_north & ~mask_equatorial, 8 + torch.floor_divide(iphi_south - 1, iring_south),
                             torch.zeros_like(pix)).long()
    irt_south = 4 * nside - iring_south - (jrll[face_south] * nside) + 1
    ipt_south = 2 * iphi_south - jpll[face_south] * iring_south - 1
    ipt_south = torch.where(ipt_south >= 2 * nside, ipt_south - 8 * nside, ipt_south)

    x_result = torch.where(mask_north, torch.bitwise_right_shift(ipt_north - irt_north, 1),
                           torch.where(mask_equatorial, torch.bitwise_right_shift(ipt_equatorial - irt_equatorial, 1),
                                       torch.where(~mask_north & ~mask_equatorial,
                                                   torch.bitwise_right_shift(ipt_south - irt_south, 1),
                                                   torch.zeros_like(pix))))
    y_result = torch.where(mask_north, torch.bitwise_right_shift(-(ipt_north + irt_north), 1),
                           torch.where(mask_equatorial,
                                       torch.bitwise_right_shift(-(ipt_equatorial + irt_equatorial), 1),
                                       torch.where(~mask_north & ~mask_equatorial,
                                                   torch.bitwise_right_shift(-(ipt_south + irt_south), 1),
                                                   torch.zeros_like(pix))))
    face_result = torch.where(mask_north, face_north,
                              torch.where(mask_equatorial, face_equatorial,
                                          torch.where(~mask_north & ~mask_equatorial, face_south,
                                                      torch.zeros_like(pix))))

    return HPD(x_result, y_result, face_result)


# Convert from ring to nested
def ring2nest(nside, ipring):
    return hpd2nest(nside, ring2hpd(nside, ipring))


# Convert angle to vector
def ang2vec(theta, phi):
    if torch.any(theta < 0.0) or torch.any(theta > torch.pi):
        raise ValueError("theta out of range [0., PI]")

    stheta = torch.sin(theta)
    x = stheta * torch.cos(phi)
    y = stheta * torch.sin(phi)
    z = torch.cos(theta)

    v = torch.stack([x, y, z], dim=1)
    return v


# Convert vector to angle
def vec2ang(v):
    theta = torch.atan2(torch.hypot(v[:, 0], v[:, 1]), v[:, 2])
    phi = torch.atan2(v[:, 1], v[:, 0])
    return torch.stack([theta, phi], dim=1)


if __name__ == "__main__":
    # TESTING
    import healpy as hp
    import numpy as np

    torch.set_default_dtype(torch.float64)
    nside_testing = 512

    # Comparison of ang2pix
    theta_vec = torch.linspace(0.0, np.pi, 10)
    phi_vec = torch.linspace(0.0, 2.0 * np.pi, 10)
    theta, phi = torch.meshgrid(theta_vec, phi_vec, indexing="ij")
    theta, phi = theta.flatten(), phi.flatten()

    out_torch = ang2pix(nside_testing, theta, phi)
    out_hp = hp.ang2pix(nside_testing, theta.numpy(), phi.numpy(), nest=False, lonlat=False)
    assert torch.all(out_torch == torch.from_numpy(out_hp)), "ang2pix does not agree with healpy.ang2pix!"

    # Comparison of pix2ang
    out_torch = pix2ang(torch.arange(hp.nside2npix(nside_testing)), nside_testing)
    out_hp = hp.pix2ang(nside_testing, np.arange(hp.nside2npix(nside_testing)), nest=False, lonlat=False)
    print("Max. difference in theta: ", torch.max(torch.abs(out_torch[:, 0] - out_hp[0])))
    print("Max. difference in phi: ", torch.max(torch.abs(out_torch[:, 1] - out_hp[1])))
    print(np.asarray(out_torch[:10, 0]), out_hp[0][:10], sep="\n")

    # Comparison of nest2ring
    out_torch = nest2ring(nside_testing, torch.arange(hp.nside2npix(nside_testing)))
    out_hp = hp.nest2ring(nside_testing, np.arange(hp.nside2npix(nside_testing)))
    assert torch.all(out_torch == torch.from_numpy(out_hp)), "nest2ring does not agree with healpy.nest2ring!"

    # Comparison of ring2nest
    out_torch = ring2nest(nside_testing, torch.arange(hp.nside2npix(nside_testing)))
    out_hp = hp.ring2nest(nside_testing, np.arange(hp.nside2npix(nside_testing)))
    assert torch.all(out_torch == torch.from_numpy(out_hp)), "ring2nest does not agree with healpy.ring2nest!"

    # Comparison of ang2vec
    out_torch = ang2vec(theta, phi)
    out_hp = hp.ang2vec(theta.numpy(), phi.numpy())
    print("Max. difference in x: ", torch.max(torch.abs(out_torch[0] - out_hp[0])))
    print("Max. difference in y: ", torch.max(torch.abs(out_torch[1] - out_hp[1])))

    # Comparison of vec2ang
    out_torch = vec2ang(out_torch)
    out_hp = hp.vec2ang(out_hp)
    print("Max. difference in theta: ", torch.max(torch.abs(out_torch[:, 0] - out_hp[0])))
    print("Max. difference in phi: ", torch.max(torch.abs(out_torch[:, 1] - out_hp[1])))
