# RL_META

DIAYN

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ step \ one \ step \ in \ environment$$

$$3. \ compute \ q_\phi(z | s_t) \ from \ discriminator \ q_\phi$$

$$4. \ improve \ value \ log(q_\phi(z | s_t)) - log(p(z)) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ log(q_\phi(z | s_t)) - log(p(z))$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


* * *
REPO
https://arxiv.org/pdf/1802.06070.pdf
