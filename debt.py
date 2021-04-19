import math


def relative_to_dollar_conversion(rate_structure, basis):
    # Converts relative rate structure to dollar structure, to allow for calculation
    converted_structure = []
    for row in rate_structure:
        converted_structure.append([row[0] * basis, row[1] * basis, row[2]])

    return converted_structure


class Debt:
    def __init__(self, rate_structure = ((0, 1000, 0.02)),
                 rate_structure_type='dollar', initial_debt=1000):
        self.debt_amount = initial_debt
        self.dollar_rate_structure = rate_structure if rate_structure_type == 'dollar'\
            else relative_to_dollar_conversion(rate_structure, initial_debt)

    def change_rate_structure(self, rate_structure, rate_structure_type):
        self.dollar_rate_structure = rate_structure if rate_structure_type == 'dollar'\
            else relative_to_dollar_conversion(rate_structure, self.debt_amount)

    def add_debt(self, debt_amount):
        self.debt_amount += debt_amount

    def prepayment(self, prepayment_amount):
        self.debt_amount -= min(prepayment_amount, self.debt_amount)


    def calculate_interest(self, basis="", monthly=True, deduction=0.206):
        basis = self.debt_amount if basis == "" else basis
        interest_bill = 0
        rate_structure = self.dollar_rate_structure

        # Make sure entire debt is covered by rate structure
        rate_structure[-1][1] = self.debt_amount

        for row in rate_structure:
            basis -= row[0]
            interest_bill += min(row[1], basis) * (row[2] if not monthly
                                                   else (math.exp(row[2]/12) -1))

        return interest_bill/(1+deduction)



if __name__ == '__main__':
    SU = Debt(rate_structure=[[0, .4, 0.02], [.4, 1, 0.03]],
              rate_structure_type='relative', initial_debt=1000)
    print(SU.calculate_interest())
    print(SU.calculate_interest())
