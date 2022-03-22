# Basics

## InitialResidualBlock

|  #   |    Layer     |    Input size     | Input source |    Output size    | b_in | b_out |
| :--: | :----------: | :---------------: | :----------: | :---------------: | :--: | :---: |
|  0   |    Input     |         -         |      -       |    C x 2b x 2b    |  -   |   b   |
|  1   | $ S^2 $ Conv |    C x 2b x 2b    |      0       |    C x 2b x 2b    |  32  |  32   |
|  2   | $SO(3)$ Conv | 50 x 64 x 64 x 64 |      1       | 50 x 64 x 64 x 64 |  32  |  32   |







# ModelCaps

|  #   |           Layer            |       Input size       | Input source |      Output size       | b_in | b_out | Capsule number | Capsule dimension |
| :--: | :------------------------: | :--------------------: | :----------: | :--------------------: | :--: | :---: | :------------: | :---------------: |
|  0   |           Input            |           -            |      -       |      6 x 64 x 64       |  -   |  32   |       -        |         -         |
|  1   |        $ S^2 $ Conv        |      6 x 64 x 64       |      0       |   50 x 64 x 64 x 64    |  32  |  32   |       -        |         -         |
|  2   |        $SO(3)$ Conv        |   50 x 64 x 64 x 64    |      1       |   50 x 64 x 64 x 64    |  32  |  32   |       -        |         -         |
|  3   |        $ S^2 $ Conv        |      6 x 64 x 64       |      0       |   50 x 64 x 64 x 64    |  32  |  32   |       -        |         -         |
|  4   |            Add             |   50 x 64 x 64 x 64    |    2 & 3     |   50 x 64 x 64 x 64    |  32  |  32   |       -        |         -         |
|  5   |        $SO(3)$ Conv        |   50 x 64 x 64 x 64    |      4       |   50 x 44 x 44 x 44    |  32  |  22   |       -        |         -         |
|  6   |        $SO(3)$ Conv        |   50 x 44 x 44 x 44    |      5       |   50 x 44 x 44 x 44    |  22  |  22   |       -        |         -         |
|  7   |        $SO(3)$ Conv        |   50 x 64 x 64 x 64    |      4       |   50 x 44 x 44 x 44    |  32  |  22   |       -        |         -         |
|  8   |      Add and reshape       |   50 x 44 x 44 x 44    |    6 & 7     | 5 x 10 x 44 x 44 x 44  |  22  |  22   |       5        |        10         |
|  9   |        Capsule Conv        | 5 x 10 x 44 x 44 x 44  |      8       | 5 x 10 x 44 x 44 x 44  |  22  |  22   |       5        |        10         |
|  10  |        Capsule Conv        | 5 x 10 x 44 x 44 x 44  |      9       | 5 x 10 x 14 x 14 x 14  |  22  |   7   |       5        |        10         |
|  11  |        Capsule Conv        | 5 x 10 x 14 x 14 x 14  |      10      | 5 x 10 x 14 x 14 x 14  |  7   |   7   |       5        |        10         |
|  12  |        Capsule Conv        | 5 x 10 x 14 x 14 x 14  |      11      | 10 x 10 x 14 x 14 x 14 |  7   |   7   |       10       |        10         |
|  13  | $SO(3)$ Integrate and norm | 10 x 10 x 14 x 14 x 14 |      12      |        10 x 10         |  7   |   -   |       10       |        10         |



# SMNIST

Deeper: 41b9b8a1364d5ce5aec6ae3d57b6b0bee03b88a0

Shorter: fbace1b686ef094027be314bb3b03993f06c3c77

> Note:
>
> 1. In residual block, out_dim_hidden = out_dim

- 1-4 : conv block1
- 5-8: conv block2



| # |      Layer      |      Input size       | Input source |      Output size      | b_in | b_out | Capsule number | Capsule dimension |
| :---: | :-----------------: | :-------------------: | :----------: | :-------------------: | :--: | :---: | :------------: | :---------------: |
|   0   |        Input        |           -           |      -       |      1 x 60 x 60      |  -   |  30   | - | - |
|   1   |    $ S^2 $ Conv     |      1 x 60 x 60      |      0       |      40x20x20x20      |  30  |  10   | - | - |
|   2   |    $SO(3)$ Conv     |      20x20x20x20      |      1       |      40x12x12x12      |  10  |   6   | - | - |
|   3   |    $ S^2 $ Conv     |      1 x 60 x 60      |      0       |      40x12x12x12      |  30  |   6   | - | - |
|   4   |         Add         |      40x12x12x12      |    2 & 3     |      40x12x12x12      |  6   |   6   | - | - |
|   5   |    $SO(3)$ Conv     |      40x12x12x12      |      4       |      50x12x12x12      |  6   |   6   | - | - |
|   6   |    $SO(3)$ Conv     |      50x12x12x12      |      5       |      50x12x12x12      |  6   |   6   | - | - |
|   7   | $SO(3)$ Conv |      40x12x12x12      |      4       |      50x12x12x12      |  6   |   6   | - | - |
|   8   |   Add and reshape   |      50x12x12x12      |    6 & 7     | 5 x 10 x 12 x 12 x 12 |  6   |   6   |       5        |        10         |
|   9   |    Capsule Conv     | 5 x 10 x 12 x 12 x 12 |      8       |  5 x 10 x 12 x 12 x 12  |  6   |   6   |       5       |        10        |
|  10  |        Capsule Conv        | 5 x 10 x 12 x 12 x 12 |      9       |  5 x 10 x 8 x 8 x 8   |  6   |   4   |       5        |        10         |
| 11 | Capsule Conv | 5 x 10 x 8 x 8 x 8 | 10 | 5 x 10 x 4 x 4 x 4 | 4 | 2 | 5 | 10 |
|  12  |        Capsule Conv        |  5 x 10 x 4 x 4 x 4   |      11      |  10 x 16 x 4 x 4 x 4  |  2   |   2   |       10       |        16         |
| 13 | $SO(3)$ Integrate and norm | 10 x 16 x 4 x 4 x 4 | 12 | 10x16 | 2 | - | 10 | 16 |



# MultiInputCapsModel

