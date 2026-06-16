# Q(k) block for H_chiral

With basis (A0, A1 | B2, B3),

H_chiral(k) = [[0, Q(k)], [Q(k)^dagger, 0]]

Q(k) = [[ t,                 v + w*exp(-i*kx) ],
        [ v + w*exp(-i*ky),  t               ]]

Explicit 4x4 H_chiral(k):
[[0, 0, t, v+w*exp(-i*kx)],
 [0, 0, v+w*exp(-i*ky), t],
 [t, v+w*exp(+i*ky), 0, 0],
 [v+w*exp(+i*kx), t, 0, 0]]
