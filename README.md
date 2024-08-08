A equação que está sendo resolvida é a equação de Poisson em duas dimensões, com condições de contorno de Dirichlet. 

**Equação de Poisson:**

```math
-\nabla^2 u(x,y) = f(x,y) \quad \text{em} \quad \Omega = (0,1) \times (0,1)
```

onde:

$u(x,y)$ é a função desconhecida que estamos resolvendo
$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ é o operador Laplaciano
$f(x,y)$ é a função de fonte (no nosso caso, $f(x,y) =  2 * \pi^2 * \operatorname{sen}(\pi * x) * \operatorname{sen}(\pi * y)$)

**Condições de contorno de Dirichlet:**

```math
u(x,y) = 0 \quad \text{em} \quad \partial\Omega
```

onde $\partial\Omega$ representa a fronteira do domínio, que neste caso são os quatro lados do quadrado unitário:

```math
\begin{align*}
u(0,y) &= 0 \quad \text{para} \quad 0 \leq y \leq 1 \\
u(1,y) &= 0 \quad \text{para} \quad 0 \leq y \leq 1 \\
u(x,0) &= 0 \quad \text{para} \quad 0 \leq x \leq 1 \\
u(x,1) &= 0 \quad \text{para} \quad 0 \leq x \leq 1
\end{align*}
```

Esta equação, com estas condições de contorno, descreve um problema de valor de fronteira em que estamos buscando uma função $u(x,y)$ que satisfaça a equação de Poisson no interior do domínio quadrado unitário, e que seja zero em todas as fronteiras deste quadrado.

**Solução Analítica:**

Um possível solução analítica  é  dada por:

```math
u(x,y) = \operatorname{sen}(\pi*x) * \operatorname{sen}(\pi*y))
```
