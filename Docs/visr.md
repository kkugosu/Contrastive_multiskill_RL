## VISR

this paper learns skills from "state action pair"

$$1. \ sample \ skill \ from \ p(z)$$

$$2. \ step \ one \ step \ in \ environment$$

$$3. \ compute \ q_\phi(z | s, a) \ from \ discriminator \ q_\phi $$

$$4. \ improve \ value \ q_\phi(z | s, a) \ by \ update \ \phi$$

$$5. \ set \ reward \ as \ q_\phi(z | s, a)$$

$$6. \ update \ sac \ with \ policy \ z \ to \ maximize \ reward$$

