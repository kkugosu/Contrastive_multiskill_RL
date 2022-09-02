
## APS

we embed state to abstract state and maximize entropy from state distribution and s,a marginal distribution

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ while \ memory \ become \ full$$

$$3. \ learn \ state \ embedding \ from \ key \ and \ query \ network$$

$$3. \ compute \ q_\phi(z | s, a) \ from \ discriminator \ q_\phi$$

$$4. \ improve \ value \ log(q_\phi(z | s, a)) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ log(q_\phi(z | s, a)) + entropy(s))$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$
