# from Márton Juhász github repository

import numpy as np


class CG:
    # Clebsch-Gordan coefficient class containg all important parameters and a function to check if the m values are in range
    def __init__(self, j1: float, m1: float, j2: float, m2: float, J: float, M: float):

        self.j1 = j1
        self.m1 = m1
        self.j2 = j2
        self.m2 = m2
        self.J = J
        self.M = M

        self.Value = 0

    def __repr__(self) -> str:
        return f"C({self.j1} {self.m1} {self.j2} {self.m2}|{self.J},{self.M}) = {self.Value}"

    def IsRangeValid(self) -> bool:
        return (self.m2 >= -self.j2 and self.m1 >= -self.j1 and self.M >= -self.J) and (
            self.m2 <= self.j2 and self.m1 <= self.j1 and self.M <= self.J
        )


def ClebschGordan(j1: float, j2: float, J: float):
    # The main algorithm that receives the starting parameters
    # The output is the non-zero Clebsch-Gordan coefficients
    if J < abs(j1 - j2) or J > j1 + j2:
        raise Exception("Paramteres does not satisfy |j1-j2|<=J<=j1+j2")

    M = np.arange(-J, J + 1, 1)[::-1]
    m1 = np.arange(-j1, j1 + 1, 1)[::-1]
    m2 = np.arange(-j2, j2 + 1, 1)[::-1]

    cglist = CreateInitialCGList(j1, m1, j2, m2, J)

    for curr_M in M:
        for curr_m1 in m1:
            for curr_m2 in m2:
                cgjm = CgJM(cglist, j1, curr_m1, j2, curr_m2, J, curr_M)
                if cgjm.Value != 0:
                    cglist.append(cgjm)

    return cglist


def CreateInitialCGList(j1, m1, j2, m2, J):
    # Initial possible Clebsch-Gordan coefficients in the J = M state that satisfies the condition that m1 + m2 = M
    cglist: list[CG] = []

    for curr_m1 in m1:
        for curr_m2 in m2:
            if curr_m1 + curr_m2 == J:
                cglist.append(CG(j1, curr_m1, j2, curr_m2, J, J))

    CalcInitialValues(cglist)

    return cglist


def CalcInitialValues(cglist: list[CG]):
    # Calculation of the initial Clebsch-Gordan coefficiant using the initiak recursion relation
    ss: list[float] = [1]

    for i, cg in enumerate(cglist):
        s = DivCalc(cg.j1, cg.m1, cg.j2, cg.m2, cg.J) * ss[i]
        ss.append(s)

    sums = sum(ss[1:])

    cglist[0].Value = np.sqrt(1 / (1 + sums))

    for i in range(1, len(cglist)):
        cg = cglist[i]
        pcg = cglist[i - 1]
        cg.Value = pcg.Value * -np.sqrt(DivCalc(pcg.j1, pcg.m1, pcg.j2, pcg.m2, pcg.J))


def DivCalc(j1, m1, j2, m2, J):
    # Function to calculate the fraction of the ladder coefficients
    m2 = m2 + 1
    a = j2 * (j2 + 1) - m2 * (m2 - 1)
    b = j1 * (j1 + 1) - m1 * (m1 - 1)
    return 0 if b == 0 else a / b


def CgJM(l: list[CG], j1, m1, j2, m2, J, M) -> CG:
    # Function to calculate the M = J-1... states by using the lower-sign recursion relation
    cg_m = CG(j1, m1, j2, m2, J, M - 1)

    if cg_m.IsRangeValid():

        cg_m1_value = SelectCG(l, j1, m1 + 1, j2, m2, J, M)
        cg_m2_value = SelectCG(l, j1, m1, j2, m2 + 1, J, M)

        cg_m.Value = (
            np.sqrt(j1 * (j1 + 1) - m1 * (m1 + 1)) / np.sqrt(J * (J + 1) - M * (M - 1))
        ) * cg_m1_value + (
            np.sqrt(j2 * (j2 + 1) - m2 * (m2 + 1)) / np.sqrt(J * (J + 1) - M * (M - 1))
        ) * cg_m2_value

    return cg_m


def SelectCG(l: list[CG], j1, m1, j2, m2, J, M) -> float:

    for cg in l:
        if (
            cg.J == J
            and cg.j1 == j1
            and cg.j2 == j2
            and cg.M == M
            and cg.m1 == m1
            and cg.m2 == m2
        ):
            return cg.Value

    return 0.0
