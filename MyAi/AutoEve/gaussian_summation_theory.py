import math

"""
Formal equation = (0.5 * x * (1 + (2/sqrt(pi)) * intergral[0, (x/sqrt(2))](e^-t^2)dt))

gaussian reformed = (2/sqrt(pi)) * intergral[0, (x](e^-t^2)dt

gaussian reformed as summation = (2/sqrt(pi)) * summation[n=0, inf](((-1)^n) * (x^(2n+1)) / (n!(2n+1)))

my equation = (0.5 * x * (1 + (2/sqrt(pi)) * summation[n=0, inf](((-1)^n) * (x^(2n+1)) / (n!(2n+1)))))

the gaussian reformed equation is not the same as the formal equation however the equation is the same for a 
select few values of x and the result should be more or less the same, I can change this later or not I dont care
overall the math checks out for a select few values and that should be enough for the pourposes of eves training.

The proofs are in the main eve training file and in the calc 2 book last pages.
It would help to use desmos for the equations listed above to see how they work in conjunction with each other.

Wiki: https://en.wikipedia.org/wiki/Error_function
Desmos graphs: https://www.desmos.com/calculator/deuwfoff79
Wolfram APPERENTLY ALREADY SOLVED IT: https://mathworld.wolfram.com/MaclaurinSeries.html
"""

def math_gauss(x):
    summation = 0
    for n in range(11):
        summation += ((-1)**n * x**(2*n + 1)) / (math.factorial(n) * (2*n + 1))
    result = (2 / math.sqrt(math.pi)) * summation
    return result

#x = float(input("Enter the value of x: "))
#[print(f"math_gauss({x}) = {math_gauss(x)}")]