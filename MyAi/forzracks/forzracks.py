import numpy as np
import asyncio
import os
import json

class Stock:
    def __init__(self, name, mu=0.1, sigma=0.2, S0=100, dt=1/252):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.price = S0
        self.dt = dt

    def step(self):
        dW = np.random.normal() * np.sqrt(self.dt)
        self.price *= np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW)
        return self.price

class Data:
    def load():
        if os.path.exists(os.path.join("stocks.json")):
            with open(os.path.join("stocks.json"), 'r') as f:
                return json.load(f)
        else:
            return None
        
    def save(data=None):
        if data is not None:
            stocks = data

        with open(os.path.join("stocks.json"), 'w') as f:
            json.dump(stocks, f)

async def run_market():
    init_stocks = Data.load()
    if init_stocks is None:
        init_stocks = {"Nvidia": 100, "Apple": 100, "Microsoft": 100, "Steam": 100}
        
    stocks = {}

    for name in init_stocks.keys():
        stocks[name] = Stock(name, S0=init_stocks[name])
    
    while True:
        for stock in stocks.values():
            init_stocks[stock.name] = stock.step()
        Data.save(init_stocks)
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run_market())