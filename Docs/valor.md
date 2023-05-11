
## VALOR

this paper learns skills from trajectary using "lstm"

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ simulate \ in \ environment \ until \ memory \ become \ full $$

$$3. \ compute \ q_\phi(z | \tau) \ from \ discriminator \ q_\phi, use \ q_\phi \ as \ lstm $$

$$4. \ improve \ value \ q_\phi(z | \tau) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(z | \tau)$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ the \ reward$$
