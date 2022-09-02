## DADS

this paper learns skills from "initial state and last state pair"

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ step \ one \ step \ in \ environment$$

$$3. \ compute \ q_\phi(s' | s, z) \ from \ discriminator \ q_\phi $$

$$4. \ improve \ value \ q_\phi(s' | s, z) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(s' | s, z)$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$


