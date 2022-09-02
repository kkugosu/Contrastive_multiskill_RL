
## APS

we embed state to abstract state and maximize entropy from state distribution and s,a marginal distribution

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ step \ one \ step \ in \ environment$$

$$3. \ compute \ q_\phi(z | s_t) \ from \ discriminator \ q_\phi$$

$$4. \ improve \ value \ log(q_\phi(z | s_t)) - log(p(z)) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ log(q_\phi(z | s_t)) - log(p(z))$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$
