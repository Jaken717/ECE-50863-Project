from ucimlrepo import fetch_ucirepo

# fetch dataset
phishing_websites = fetch_ucirepo(id=327)

# data (as pandas dataframes)
X = phishing_websites.data.features
y = phishing_websites.data.targets

# metadata
print(phishing_websites.metadata)

# variable information
print(phishing_websites.variables)

# save a local copy so we don't re-download every time
X.join(y).to_csv("data/phishing_websites.csv", index=False)
print("Saved data/phishing_websites.csv", X.shape, y.shape)
