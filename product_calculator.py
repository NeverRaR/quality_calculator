from typing import List
import numpy as np

quality_mod_effect = 0.062
production_mod_effect = 0.25


class Machine:
    mod_slot_num = 4
    basic_production_effect = 1
    production_mod_num = 0
    quality_matrix = []

    def __init__(self, mod_slot_num, basic_production_effect, production_mod_num):
        self.mod_slot_num = mod_slot_num
        self.basic_production_effect = basic_production_effect
        self.production_mod_num = production_mod_num
        self.update_quality_matrix()

    """
    Common P * (1-Q)
    Uncommon 0.9 * P * Q
    Rare 0.09 * P * Q
    Epic 0.009 * P * Q
    Legendary 0.001 * P * Q
    """

    def update_quality_matrix(self):
        quality_effect = (
            self.mod_slot_num - self.production_mod_num
        ) * quality_mod_effect
        production_effect = (
            production_mod_effect * self.production_mod_num
            + self.basic_production_effect
        )
        self.quality_matrix = np.array(
            [
                [
                    production_effect * (1 - quality_effect),
                    production_effect * quality_effect * 0.9,
                    production_effect * quality_effect * 0.09,
                    production_effect * quality_effect * 0.009,
                    production_effect * quality_effect * 0.001,
                ],
                [
                    0,
                    production_effect * (1 - quality_effect),
                    production_effect * quality_effect * 0.9,
                    production_effect * quality_effect * 0.09,
                    production_effect * quality_effect * 0.01,
                ],
                [
                    0,
                    0,
                    production_effect * (1 - quality_effect),
                    production_effect * quality_effect * 0.9,
                    production_effect * quality_effect * 0.1,
                ],
                [
                    0,
                    0,
                    0,
                    production_effect * (1 - quality_effect),
                    production_effect * quality_effect,
                ],
                [0, 0, 0, 0, production_effect],
            ]
        )


class AssemblingMachine(Machine):
    def __init__(self, production_mod_num):
        super().__init__(4, 1, production_mod_num)


class EletricFurance(Machine):
    def __init__(self, production_mod_num):
        super().__init__(2, 1, production_mod_num)


class EletronmagneticPlant(Machine):
    def __init__(self, production_mod_num):
        super().__init__(5, 1.5, production_mod_num)


class Foundry(Machine):
    def __init__(self, production_mod_num):
        super().__init__(4, 1.5, production_mod_num)


class CryogenicPlant(Machine):
    def __init__(self, production_mod_num):
        super().__init__(8, 1, production_mod_num)


class Recycler(Machine):
    def __init__(self):
        super().__init__(4, 0.25, 0)

    def update_quality_matrix(self):
        super().update_quality_matrix()
        self.quality_matrix[4][4] = 1


class ProductionMatrix:
    chains: List[List[Machine]] = []

    def __init__(self, chains: List[List[Machine]]):
        self.chains = chains

    def update(self, input_quality_level: int):
        cur_col_idx = 0
        cur_machine = self.chains[input_quality_level][cur_col_idx]
        while cur_machine.production_mod_num == cur_machine.mod_slot_num:
            cur_col_idx = cur_col_idx + 1
            if cur_col_idx == len(self.chains[input_quality_level]):
                for m in self.chains[input_quality_level]:
                    m.production_mod_num = 0
                    m.update_quality_matrix()
                return False
            cur_machine.production_mod_num = 0
            cur_machine.update_quality_matrix()
            cur_machine = self.chains[input_quality_level][cur_col_idx]
        cur_machine.production_mod_num = cur_machine.production_mod_num + 1
        cur_machine.update_quality_matrix()
        return True

    def set_mod_config(self, config: List[int], input_quality_level: int):
        for i in range(0, len(self.chains[input_quality_level])):
            cur_machine = self.chains[input_quality_level][i]
            cur_machine.production_mod_num = config[input_quality_level][i]
            cur_machine.update_quality_matrix()

    def get_mod_config(self):
        result = []
        for chain in self.chains:
            line_config = []
            for m in chain:
                line_config.append(m.production_mod_num)
            result.append(line_config)
        return result


def quality_calculation(machine_matrix: ProductionMatrix, input_quality_level: int):
    machine_chains = machine_matrix.chains
    convert_quality_matrix = np.eye(5)
    cycle_quality_matrix = np.eye(5)
    for i in range(0, 5):
        single_convert_quality_matrix = np.eye(5)
        idx = 0
        for m in machine_chains[i]:
            if idx + 1 == len(machine_chains[i]):
                break
            single_convert_quality_matrix = np.matmul(
                single_convert_quality_matrix, m.quality_matrix
            )
            idx = idx + 1
        convert_quality_matrix[i] = single_convert_quality_matrix[i]
    input = np.array([0, 0, 0, 0, 0])
    input[input_quality_level] = 1
    input = np.matmul(input, convert_quality_matrix)
    for i in range(0, 5):
        cycle_quality_matrix[i] = machine_chains[i][
            len(machine_chains[i]) - 1
        ].quality_matrix[i]
    legendary_count = 0.0
    craft_count = 0
    recycler = Recycler()
    while True:
        next = np.matmul(input, cycle_quality_matrix)
        legendary_count = legendary_count + next[4]
        next[4] = 0
        next = np.matmul(next, recycler.quality_matrix)
        if np.abs(input - next).max() < 0.000000001:
            break
        input = next
        craft_count = craft_count + 1
    return legendary_count


def select_bast_mod_config(machine_matrix: ProductionMatrix, input_quality_level: int):
    max_legendary_count = quality_calculation(machine_matrix, input_quality_level)
    best_mod_config = machine_matrix.get_mod_config()
    while machine_matrix.update(input_quality_level):
        legendary_count = quality_calculation(machine_matrix, input_quality_level)
        if max_legendary_count < legendary_count:
            max_legendary_count = legendary_count
            best_mod_config = machine_matrix.get_mod_config()
    return best_mod_config, max_legendary_count


np.set_printoptions(suppress=True)
# [Foundry(), EletricFurance(), Foundry(), AssemblingMachine()],
# [EletronmagneticPlant()],
# [AssemblingMachine()],
production_chains = [
    [AssemblingMachine(0)],
    [AssemblingMachine(0)],
    [AssemblingMachine(0)],
    [AssemblingMachine(0)],
    [AssemblingMachine(0)],
]

machine_matrix = ProductionMatrix(production_chains)
best_mod_config = []
max_legendary_count = 0
legendary_count = quality_calculation(machine_matrix, 0)
print(legendary_count)
for i in range(0, 5):
    (best_mod_config, max_legendary_count) = select_bast_mod_config(
        machine_matrix, 4 - i
    )
    machine_matrix.set_mod_config(best_mod_config, 4 - i)

print(best_mod_config)
print(max_legendary_count)