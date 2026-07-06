# PolyNEAT

Biblioteka Pythonowa do **algorytmów neuroewolucji**. Pierwszy zaimplementowany algorytm to klasyczny NEAT (Stanley & Miikkulainen, 2002). Architektura jest zaprojektowana tak, by bez przebudowy rdzenia można było dodawać warianty NEAT (HyperNEAT, ES-HyperNEAT, NEAT-LSTM) oraz metody głębokiej neuroewolucji.

---

## Instalacja

Wymaga Pythona 3.11+. Zalecany menedżer pakietów to [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Alternatywnie ze standardowym `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

---

## Szybki start — benchmark XOR

```bash
uv run python examples/xor_baseline.py
```

Artefakty (topologie sieci, najlepszy genom w JSON i pickle) trafiają do `examples/xor_artifacts/`.

---

## Struktura projektu

```
poly-neat/
├── polyneat/                  główny pakiet biblioteki
│   ├── __init__.py            publiczne API — wszystko co eksportujemy
│   ├── config/                konfiguracja eksperymentów
│   ├── core/                  protokoły i typy danych
│   ├── algorithms/
│   │   └── neat/              implementacja NEAT + mutacje
│   ├── nn/                    funkcje aktywacji + narzędzia topologiczne
│   ├── evaluators/            ocenianie fenotypów (sekwencyjne, równoległe, XOR)
│   ├── runner/                pętla ewolucji, callbacki, kryteria stopu
│   ├── logging_utils/         własny, kolorowany logger
│   ├── viz/                   wizualizacja topologii sieci
│   └── utils/                 pomocnicze (RNG, serializacja)
├── examples/
│   ├── xor_baseline.py        skrypt demonstracyjny — NEAT na XOR
│   └── xor_baseline.yaml      konfiguracja eksperymentu XOR
└── docs/                      dokumenty projektowe
```

---

## Za co odpowiada każdy moduł

### `polyneat/config/`

| Plik | Odpowiedzialność |
|---|---|
| `algorithm_config.py` | Bazowy dataclass `AlgorithmConfig` — parametry wspólne dla każdego algorytmu (`population_size`, `number_of_input_nodes`, `random_seed` itp.). Zawiera `load_from_yaml_file` i `from_dict` (ścisłe — nieznane klucze rzucają `ConfigurationError`). |
| `neat_config.py` | `NEATConfig` dziedziczy z `AlgorithmConfig` i dodaje wszystkie hiperparametry specyficzne dla NEAT: prawdopodobieństwa mutacji, współczynniki zgodności gatunkowej, parametry selekcji. |
| `configuration_errors.py` | `ConfigurationError` — rzucany gdy konfiguracja jest nieprawidłowa; komunikat zawsze podaje pole, wartość i powód. |

### `polyneat/core/`

Serce biblioteki. Definiuje **protokoły** (interfejsy w stylu Go/Rust) które każdy algorytm musi zaimplementować.

| Plik | Odpowiedzialność |
|---|---|
| `component_protocols.py` | Dziesięć `@runtime_checkable Protocol`: `Genome`, `Phenotype`, `PhenotypeBuilder`, `MutationOperator`, `CrossoverOperator`, `ParentSelection`, `Speciator`, `FitnessEvaluator`, `NeuroevolutionAlgorithm`, `InnovationTracker`. Żaden z nich nie jest klasą bazową — wystarczy zaimplementować odpowiednie metody. |
| `population.py` | Zamrożony dataclass `Population(genomes, species_assignments, generation_number)`. Niezmienny — każde pokolenie tworzy nowy obiekt. |
| `generation_statistics.py` | Zamrożony dataclass `GenerationStatistics` — statystyki jednego pokolenia (najlepsza i średnia fitness, liczba gatunków, czas). |
| `type_aliases.py` | Aliasy typów: `FitnessValue = float`, `InnovationId = int`, `SpeciesId = int`. Poprawiają czytelność sygnatur. |

### `polyneat/algorithms/neat/`

Implementacja klasycznego NEAT. Każdy aspekt algorytmu żyje we własnym pliku.

| Plik | Odpowiedzialność |
|---|---|
| `neat_genome.py` | Zamrożone dataclassy `NodeGene` i `ConnectionGene`, oraz `NEATGenome`. Genomu nie można modyfikować w miejscu — mutacja zawsze zwraca nowy obiekt. `__post_init__` waliduje brak duplikatów węzłów i innowacji. |
| `global_innovation_tracker.py` | Przydziela globalne `InnovationId` dla nowych połączeń. W obrębie jednego pokolenia ta sama para `(źródło, cel)` dostaje ten sam ID (deduplikacja), dzięki czemu krzyżowanie może wyrównać geny o identycznej strukturze. Po każdym pokoleniu tabela deduplikacji jest czyszczona, ale licznik nigdy nie jest resetowany. |
| `mutations/add_node_mutation.py` | Losuje jedno aktywne połączenie, wyłącza je, wstawia nowy węzeł ukryty, dodaje dwa nowe połączenia: wejście→węzeł (waga 1.0) i węzeł→wyjście (oryginalna waga). |
| `mutations/add_connection_mutation.py` | Próbuje dodać nowe połączenie między nieodłączonymi węzłami. Sprawdza brak duplikatu, brak pętli własnej, brak cyklu (BFS od celu — jeżeli źródło jest osiągalne, cykl istnieje) i ograniczenia topologiczne (węzły wejściowe nie mogą być celem, węzły wyjściowe nie mogą być źródłem). |
| `mutations/weight_modification_mutation.py` | Per-połączenie: z prawdopodobieństwem `p_perturb` dodaje szum Gaussowski, z prawdopodobieństwem `p_replace` losuje wagę od nowa z zakresu inicjalizacji. |
| `mutations/toggle_connection_enabled_mutation.py` | Przełącza flagę `is_enabled` losowego połączenia. Przed ponownym włączeniem wyłączonego połączenia sprawdza, czy nie tworzy cyklu — kluczowy fix, bo `AddConnectionMutation` sprawdza cykle tylko na aktualnie aktywnych połączeniach. |
| `mutations/composite_neat_mutation.py` | Stosuje mutacje po kolei: ModyfikacjaWag → DodajPołączenie → DodajWęzeł → Przełącz. Każda mutacja jest opcjonalna (własne prawdopodobieństwo). |
| `neat_crossover.py` | Krzyżowanie wyrównane po `innovation_id`. Geny pasujące (oba rodzice mają ten sam ID) są losowo dziedziczone z prawdopodobieństwem zadanym w konfiguracji. Geny niepasujące (disjoint/excess) są brane od lepszego rodzica. Po złożeniu potomka `_resolve_enabled_connection_cycles` usuwa ewentualne cykle — mogą się pojawić gdy gene był wyłączony u jednego rodzica i ta 25% szansa na włączenie tworzy pętlę. |
| `compatibility_distance_speciator.py` | Oblicza odległość zgodności δ = c₁·E/N + c₂·D/N + c₃·W̄ (nadmiarowe/rozłączne geny + różnica wag) i przydziela genomy do gatunków na podstawie porównania z reprezentantem gatunku. Po każdej specjacji reprezentant każdego gatunku jest **losowany na nowo** spośród aktualnych członków (representative resampling — zgodnie z oryginalną pracą; zamrożony reprezentant powodował sztuczną fragmentację gatunków). |
| `tournament_parent_selection.py` | Turniej: losuje `tournament_size` osobników, zwraca najlepszego. Powtarza dla każdego żądanego rodzica. |
| `torch_feedforward_phenotype.py` | Sieć neuronowa wykonana jako `nn.Module` PyTorcha. Przy tworzeniu oblicza topologiczny porządek węzłów (algorytm Kahna), potem `forward_pass` iteruje węzły w tej kolejności, sumuje ważone wejścia i stosuje funkcję aktywacji. Obsługuje wejścia wsadowe `[batch, n_inputs]`. |
| `neat_phenotype_builder.py` | Buduje `TorchFeedForwardPhenotype` z `NEATGenome`. Przekazuje urządzenie (CPU/GPU). |
| `neat_algorithm.py` | `NEATAlgorithm` — klasa główna. `from_config()` wytwarza wszystkie komponenty z `NEATConfig`. `create_initial_population()` tworzy minimalne genomy (wejścia + bias → wyjścia). `advance_one_generation()` wykonuje pełen cykl: specjacja → adjusted fitness → alokacja potomków (proporcjonalna do **sumy** adjusted fitness gatunku) → elityzm → survival threshold (tylko najlepsze 20% gatunku zostaje rodzicami) → reprodukcja (z rzadkim krzyżowaniem międzygatunkowym, p=0.001) → reset trackera innowacji. |

### `polyneat/nn/`

| Plik | Odpowiedzialność |
|---|---|
| `activation_functions.py` | `sigmoid`, `tanh`, `relu`, `leaky_relu`, `identity` jako callable'e PyTorcha. Słownik `ACTIVATION_FUNCTION_NAME_TO_CALLABLE` + `resolve_activation_function_by_name(name)` rzucający `ConfigurationError` przy nieznanej nazwie. |
| `topology_utilities.py` | `compute_topological_order_of_node_ids` — algorytm Kahna, rzuca `ValueError` na cyklu. `would_directed_edge_create_cycle` — BFS od kandydującego celu; jeżeli źródło jest osiągalne, krawędź tworzy cykl. |

### `polyneat/evaluators/`

| Plik | Odpowiedzialność |
|---|---|
| `sequential_evaluator_base.py` | `SequentialFitnessEvaluator` — klasa bazowa dla oceniania jednego fenotypu na raz. Wystarczy nadpisać `evaluate_single_phenotype`. |
| `parallel_evaluator_wrapper.py` | `ParallelFitnessEvaluatorWrapper` opakowuje dowolny evaluator i ocenia równolegle przez `joblib`. Domyślnie `prefer="threads"`. |
| `xor_evaluator.py` | `XORFitnessEvaluator` — ocenia fenotyp na czterech wzorcach XOR. Fitness = Σ(1 − (oczekiwane − rzeczywiste)²), max = 4.0. Próg rozwiązanego XOR: ≥ 3.95. |

### `polyneat/runner/`

| Plik | Odpowiedzialność |
|---|---|
| `evolution_runner.py` | `EvolutionRunner` — główna pętla: buduje fenotypy → ocenia fitness → śledzi najlepszy genom → wywołuje `advance_one_generation` → sprawdza warunek stopu. Zwraca `EvolutionResult`. |
| `run_context.py` | `RunContext` — bieżący stan biegu (ID, czas startu, numer pokolenia, historia statystyk, najlepszy dotychczas genom). Przekazywany do wszystkich callbacków. |
| `termination_criteria.py` | `MaxGenerationsTermination`, `TargetFitnessTermination`, `FitnessStagnationTermination`, `CompositeTermination` (logika OR). |
| `evolution_callback_protocol.py` | Protokół `EvolutionCallback` z sześcioma hakami: `on_run_started`, `on_generation_started`, `on_population_evaluated`, `on_generation_completed`, `on_new_best_genome_found`, `on_run_completed`. `BaseEvolutionCallback` dostarcza puste implementacje domyślne. |
| `builtin_evolution_callbacks.py` | `ConsoleStatisticsLogger` (tabela rich), `TensorBoardLogger`, `BestGenomePersister` (JSON + pickle), `NetworkTopologyVisualizer`. |

### `polyneat/logging_utils/`

| Plik | Odpowiedzialność |
|---|---|
| `custom_logger.py` | Jedyna dozwolona ścieżka do loggera: `get_logger(__name__)`. Rejestruje `CustomLogger` jako klasę przez `logging.setLoggerClass` przy imporcie. Każdy logger sam podłącza swój handler — nie ma propagacji. |
| `colored_level_formatter.py` | `ColoredLevelFormatter` — koloruje treść wiadomości (nie cały wiersz) kolorami `colorama`. DEBUG=niebieski, INFO=zielony, WARNING=żółty, ERROR=czerwony, CRITICAL=ciemnoczerwony. |
| `logging_config.py` | `LoggingConfig` — poziom logowania, format wiadomości, opcjonalny katalog logów do pliku. |

### `polyneat/viz/`

`network_topology_renderer.py` — `render_genome_topology(genome, output_path)`. Używa `matplotlib` i `networkx`, `matplotlib.use("Agg")` zapewnia działanie bez wyświetlacza (serwer, CI). Węzły wejściowe/bias/ukryte/wyjściowe mają różne kolory; wyłączone połączenia są przerywane.

### `polyneat/utils/`

| Plik | Odpowiedzialność |
|---|---|
| `random_generator_factory.py` | `create_seeded_random_generator(seed)` — zwraca `numpy.random.Generator`. Jedna funkcja, jeden punkt kontroli ziarna. |
| `artifact_serialization.py` | `save_as_json`, `load_from_json`, `save_as_pickle`, `load_from_pickle`. |

---

## Jak działa NEAT — algorytm krok po kroku

### Skąd pochodzi implementacja

NEAT jest oparty na oryginalnej publikacji:

> Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.

Artykuł jest dostępny bezpłatnie na stronie autora. Każda decyzja implementacyjna w kodzie (formuła odległości zgodności, reguły krzyżowania, alokacja offspring) ma swoje źródło w tym dokumencie.

---

### Problem, który NEAT rozwiązuje

Klasyczna ewolucja sieci neuronowych ma trzy fundamentalne problemy:

**1. Problem konkurujących konwencji**
Jeśli ewolucja niezależnie odkrywa ten sam ukryty węzeł w dwóch genomach, strukturalnie są identyczne, ale węzły mają inne numery. Krzyżowanie takich genomów produkuje potomka z podwojonymi węzłami — bałagan.

*Rozwiązanie NEAT:* każda nowa krawędź strukturalna dostaje globalny `InnovationId`. Jeśli ta sama krawędź `(A→B)` pojawia się w kilku genomach w tym samym pokoleniu, wszyscy dostają ten sam ID. Krzyżowanie wyrównuje geny po `InnovationId`, a nie po indeksie.

**2. Problem ochrony innowacji strukturalnych**
Nowo dodany węzeł ukryty zaburza sieć — fitness spada. Bez ochrony gatunek z innowacją topologiczną zginie zanim zdąży ją zoptymalizować.

*Rozwiązanie NEAT:* specjacja. Genomy podobne strukturalnie (mała odległość zgodności) tworzą jeden gatunek. Selekcja odbywa się wewnątrz gatunków, a fitness jest dzielona przez rozmiar gatunku (shared fitness). Każdy gatunek konkuruje sam ze sobą.

**3. Problem minimalnej wymiarowości**
Sieci losowo inicjowane mają tyle samo wolności co duże sieci, ale są trudniejsze do ewolucji. Duże przestrzenie wag → wolna konwergencja.

*Rozwiązanie NEAT:* start od minimalnej sieci (wejścia + bias → wyjścia, bez węzłów ukrytych). Złożoność topologiczna rośnie stopniowo przez mutacje `AddNode` i `AddConnection`.

---

### Kodowanie genomu

```
NodeGene:
  node_id              : int       — unikalny identyfikator
  node_type            : str       — "input" | "hidden" | "output" | "bias"
  activation_function  : str       — "sigmoid" | "tanh" | "relu"

ConnectionGene:
  innovation_id        : int       — globalny numer innowacji
  source_node_id       : int
  target_node_id       : int
  weight               : float
  is_enabled           : bool      — AddNodeMutation wyłącza dzielone połączenie
```

Zarówno `NodeGene`, `ConnectionGene` jak i `NEATGenome` to **zamrożone dataclassy** — żaden z operatorów nie modyfikuje istniejącego genomu. Mutacja zawsze zwraca nowy obiekt.

---

### Cykl jednego pokolenia (`advance_one_generation`)

```
Populacja t
    │
    ▼
[1] Specjacja
    CompatibilityDistanceSpeciator porównuje każdy genom
    z reprezentantem każdego gatunku.
    δ = c₁·E/N + c₂·D/N + c₃·W̄
      E — geny nadmiarowe (excess)
      D — geny rozłączne (disjoint)
      W̄ — średnia różnica wag genów pasujących
      N — normalizacja (rozmiar większego genomu)
    Jeśli δ < threshold → ten sam gatunek.
    Po przydziale reprezentant każdego gatunku jest losowany
    na nowo spośród członków bieżącego pokolenia.
    │
    ▼
[2] Adjusted fitness
    Dla każdego osobnika i w gatunku s (rozmiar |s|):
    adjusted_fitness[i] = raw_fitness[i] / |s|
    (fitness jest "dzielona" między gatunek)
    │
    ▼
[3] Stagnacja
    Jeśli gatunek nie poprawił najlepszego raw fitness
    przez species_stagnation_generations_limit pokoleń → usuwany.
    Gatunek zawierający globalnie najlepszy genom zawsze przeżywa.
    │
    ▼
[4] Alokacja potomków
    Każdy gatunek dostaje offspring_slots proporcjonalnie do
    SUMY adjusted fitness swoich członków (= średnia raw fitness
    gatunku — jak w oryginalnej pracy). Zaokrąglanie metodą
    największych reszt, więc suma slotów = population_size.
    │
    ▼
[5] Elityzm
    Jeśli gatunek ma ≥ minimum_species_size_for_elitism członków,
    top species_elitism_count genomów przechodzi bez zmian.
    │
    ▼
[6] Reprodukcja
    Survival threshold: rodzicami może być tylko najlepsze
    species_survival_fraction_for_reproduction (20%) gatunku,
    minimum 2 osobniki.
    Dla każdego wolnego slotu:
      - 75% szansy: krzyżowanie dwóch rodziców wybranych turniejem;
        z prawdopodobieństwem probability_of_interspecies_mating
        (0.001) drugi rodzic pochodzi z całej populacji
        (krzyżowanie międzygatunkowe)
      - 25% szansy: klonowanie jednego rodzica
      Wynik → mutacje kompozytowe
    │
    ▼
[7] Reset trackera innowacji (tabela deduplikacji na nowe pokolenie)
    │
    ▼
Populacja t+1
```

---

### Krzyżowanie

Krzyżowanie wyrównuje geny po `InnovationId`:

```
Rodzic 1 (lepszy): [1][2][3][ ][5][6][ ][8]
Rodzic 2 (gorszy): [1][2][ ][4][ ][6][7][ ]
                    ↑  ↑        ↑
                  pasujące   pasujące
                  (oba)       (oba)

Geny pasujące (1,2,6): losowana jest kopia od lepszego lub gorszego rodzica
Geny rozłączne (3,5,8 tylko u lepszego): dziedziczone od lepszego
Geny rozłączne (4,7 tylko u gorszego): odrzucane
```

Jeśli gen był wyłączony u któregokolwiek rodzica, potomek ma 75% szansy odziedziczyć go jako wyłączony (ochrona przed chaosem topologicznym). Po złożeniu całego genomu potomka uruchamiany jest `_resolve_enabled_connection_cycles` — usuwa ewentualne cykle powstałe przez kombinację stanu włączenia.

---

## Benchmark XOR

### Dlaczego XOR

XOR (wyłączna alternatywa) to **standardowy benchmark NEAT** z oryginalnej pracy. Jest użyteczny z kilku powodów:

- Jest **nieseparowalny liniowo** — sieć bez węzłów ukrytych nie może go rozwiązać. Algorytm musi wyewoluować odpowiednią topologię.
- Jest **trywialnie weryfikowalny** — 4 wzorce wejściowe, zero niejednoznaczności.
- Stanley i Miikkulainen użyli XOR do walidacji NEAT w 2002 roku — daje punkt odniesienia.

| Wejście | Oczekiwane wyjście |
|---|---|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

### Funkcja fitness

```python
fitness = sum(1.0 - (oczekiwane - rzeczywiste)²)   # dla każdego z 4 wzorców
```

**Maksimum = 4.0** (wszystkie wzorce bezbłędne).  
**Próg rozwiązania: ≥ 3.95**.

Używamy **błędu kwadratowego** zamiast absolutnego, bo:
- Błąd absolutny daje płaski gradient wokół lokalnego optimum "3 wzorce poprawne" (fitness = 3.0 niezależnie od tego jak bardzo sieć myli czwarty wzorzec)
- Błąd kwadratowy penalizuje mocniej duże błędy i łagodniej małe → istnieje gradient zachęcający sieć do redukowania błędu na czwartym wzorcu

Innymi słowy: sieć która daje wynik 0.5 dla wzorca (1,1) (pół błędu) dostaje fitness 3.75, a nie 3.5 jak przy błędzie absolutnym. To tworzy wyraźny sygnał do dalszej nauki.

---

### Parametry konfiguracji i uzasadnienie

Plik konfiguracyjny: `examples/xor_baseline.yaml`

#### Parametry ogólne

```yaml
population_size: 150
```
Zgodnie z oryginalną pracą Stanley'a. 150 osobników daje wystarczającą różnorodność bez nadmiernego kosztu obliczeniowego.

```yaml
number_of_input_nodes: 2
number_of_output_nodes: 1
random_seed: 42
```
XOR ma dwa wejścia (x₁, x₂) i jedno wyjście. Sieć inicjowana jest też z węzłem bias (automatycznie).

---

#### Zakresy wag

```yaml
initial_weight_range_min: -2.0
initial_weight_range_max: 2.0
weight_perturbation_strength_sigma: 0.5
```

Oryginalny artykuł używa zakresu [-1, 1] w połączeniu ze **stromym sigmoidem** (nachylenie 4.9), dzięki któremu wyjścia saturują się przy wagach zwykłej wielkości. Używamy tego samego stromego sigmoidu (patrz sekcja o funkcjach aktywacji), a nieco szerszy zakres [-2, 2] z perturbacją σ=0.5 daje ewolucji szybki dostęp do potrzebnych wartości wag.

---

#### Prawdopodobieństwa mutacji

```yaml
probability_of_add_node_mutation: 0.03
probability_of_add_connection_mutation: 0.10
probability_of_weight_perturbation: 0.80
probability_of_weight_replacement: 0.10
probability_of_toggle_connection_enabled: 0.01
```

- **AddNode = 3%** — topologia rośnie powoli. Za częste dodawanie węzłów prowadzi do wielkich, trudnych do optymalizacji sieci.
- **AddConnection = 10%** — nieco wyższe niż w papierze (tam 5%), bo XOR wymaga konkretnych połączeń do węzłów ukrytych, które muszą zostać odkryte przez ewolucję.
- **WeightPerturbation = 80%** — wagi są perturbowane w prawie każdym pokoleniu. Bez tego sieci nie uczą się nic między pokoleniami.
- **WeightReplacement = 10%** — mała szansa na całkowity reset wagi pozwala uciec z lokalnych minimów.
- **Toggle = 1%** — rzadkie, bo często prowadzi do zakłóceń topologicznych.

---

#### Specjacja

```yaml
compatibility_distance_coefficient_excess_c1: 1.0
compatibility_distance_coefficient_disjoint_c2: 1.0
compatibility_distance_coefficient_weight_difference_c3: 0.4
compatibility_distance_threshold: 3.0
```

Wartości wprost z oryginalnej pracy dla XOR. Współczynniki c₁ = c₂ = 1.0 oznaczają równe wagi dla genów nadmiarowych i rozłącznych. c₃ = 0.4 lekko obniża znaczenie różnicy wag (bo te mogą się naturalnie różnić między genami tej samej struktury).

Próg δ < 3.0 tworzy nowy gatunek. Zbyt niski próg → eksplozja liczby gatunków (każdy osobnik osobno). Zbyt wysoki → brak ochrony dla innowacji. 3.0 to wartość z publikacji, sprawdzona na wielu konfiguracjach.

---

#### Zarządzanie gatunkami

```yaml
species_elitism_count: 1
species_stagnation_generations_limit: 15
minimum_species_size_for_elitism: 5
```

- **Elityzm = 1** — najlepszy osobnik każdego gatunku (≥5 członków) przechodzi bez zmian. Zapobiega utracie odkrytych dobrych rozwiązań.
- **Stagnacja = 15** — gatunek bez poprawy najlepszego raw fitness przez 15 pokoleń jest usuwany. Ważne: stagnacja śledzi **raw fitness** (absolutną wartość), a nie adjusted fitness. Gdyby śledzić adjusted fitness, rosnący gatunek wyglądałby jakby się pogarszał (adjusted = raw / rozmiar maleje gdy rośnie mianownik) — to bug, który uniemożliwiał rozwiązanie XOR.
- **MinRozmiarElityzmu = 5** — gatunki z 1–4 osobnikami nie mają elityzmu (i tak nie ma sensu "chronić" jednego osobnika który jest jedynym przedstawicielem).

---

#### Reprodukcja i selekcja

```yaml
probability_of_crossover_vs_mutation_only: 0.75
probability_of_inheriting_from_fitter_parent_for_matching_genes: 0.50
probability_of_interspecies_mating: 0.001
tournament_size_for_parent_selection: 3
species_survival_fraction_for_reproduction: 0.2
```

- **Krzyżowanie = 75%** — większość potomków pochodzi z krzyżowania. Reszta to klony + mutacja (pozwala eksplorować bez drugiego rodzica, ważne dla małych gatunków).
- **Dziedziczenie od lepszego rodzica = 50%** — dla genów pasujących (oba rodzice mają ten sam innovation_id) równa szansa. To zgodne z papierem; inne implementacje używają 100% od lepszego, ale 50% zachowuje więcej różnorodności.
- **Krzyżowanie międzygatunkowe = 0.001** — wartość wprost z oryginalnej pracy. Rzadkie mieszanie materiału genetycznego między gatunkami; drugi rodzic wybierany jest turniejem z całej populacji.
- **Rozmiar turnieju = 3** — balans między presją selekcyjną (za mały turniej → wolna konwergencja) a różnorodnością (za duży → przedwczesna konwergencja).
- **Survival threshold = 0.2** — przed reprodukcją gatunek jest przycinany do najlepszych 20% członków (min. 2); tylko oni mogą być rodzicami. Zgodne z oryginalną implementacją Stanleya (`survival_thresh`). To najważniejszy pojedynczy parametr dla tempa konwergencji — bez niego słabe osobniki wciąż trafiały do puli rodziców i XOR wymagał ~3× więcej pokoleń.

---

#### Funkcje aktywacji

```yaml
default_activation_function_for_hidden_nodes: steepened_sigmoid
default_activation_function_for_output_nodes: steepened_sigmoid
```

`steepened_sigmoid` to φ(x) = 1/(1 + e^(−4.9x)) — **dokładnie ta funkcja, której używa oryginalna praca (sekcja 4.1)**. Nachylenie 4.9 pozwala sieci osiągać wyjścia bliskie 0/1 przy wagach zwykłej wielkości. Standardowy sigmoid (nachylenie 1) wymaga wag |w| ≥ 3–5 do saturacji, przez co ewolucja potrzebuje ~2× więcej pokoleń na XOR (zmierzone: śr. 60 vs 33 pokoleń na 10 ziarnach).

Wyjście w zakresie (0, 1) jest wygodne dla binarnych problemów jak XOR. Nowe węzły ukryte też dostają tę funkcję, choć ewolucja może wybrać inną gdy `available_activation_functions` zawiera więcej opcji.

---

### Kryterium sukcesu — uwaga o porównywalności z oryginalną pracą

Oryginalna praca uznaje sieć za rozwiązanie XOR, gdy **wszystkie cztery wyjścia są po właściwej stronie 0.5** (poprawna klasyfikacja). Nasz próg `fitness ≥ 3.95` z błędem kwadratowym jest **dużo ostrzejszy**: wymaga, by każde wyjście było średnio w odległości ~0.11 od celu. Sieć, którą praca uznałaby za rozwiązanie (np. wyjścia 0.3/0.7/0.7/0.3), ma u nas fitness zaledwie 3.64. Dlatego liczby pokoleń mierzone tymi dwoma kryteriami nie są porównywalne — raportujemy oba (pomocnik: `XORFitnessEvaluator.classifies_all_patterns_correctly`).

### Wyniki benchmarku

Przy powyższej konfiguracji NEAT rozwiązuje XOR niezawodnie:

| Ziarno (`seed`) | Najlepsza fitness | Pokoleń do fitness ≥ 3.95 | Pokoleń do kryterium z pracy |
|---:|---:|---:|---:|
| 0 | 3.9831 | 31 | 17 |
| 1 | 3.9779 | 23 | 17 |
| 2 | 3.9931 | 34 | 25 |
| 7 | 3.9945 | 39 | 29 |
| 13 | 3.9701 | 31 | 25 |
| 42 | 3.9819 | 33 | 10 |
| 100 | 3.9859 | 28 | 28 |
| 200 | 3.9600 | 26 | 15 |
| 300 | 3.9790 | 26 | 26 |
| 400 | 3.9756 | 60 | 57 |

**10/10 ziaren; średnio 33.1 pokoleń do fitness ≥ 3.95 i 24.9 pokoleń do kryterium z oryginalnej pracy (praca raportuje średnio 32).**

Poprzednia wersja implementacji potrzebowała średnio ~165 pokoleń do fitness ≥ 3.95. Przyspieszenie pochodzi z (ablacja na tych samych 10 ziarnach):

1. **Survival threshold + krzyżowanie międzygatunkowe + resampling reprezentantów + poprawiona alokacja potomków** — ~165 → 60.1 pokoleń. Dominujący wkład ma survival threshold (presja selekcyjna); sama poprawka resamplingu i alokacji nie zmienia tempa na XOR, ale usuwa fragmentację gatunków istotną w dłuższych biegach.
2. **Stromy sigmoid (4.9) zamiast standardowego** — 60.1 → 33.1 pokoleń.

---

## Używanie biblioteki

### Logger — jedyna dozwolona ścieżka

```python
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)
logger.info("Wiadomość")
logger.debug("Debug z %s", "argumentem")
```

Nigdy `logging.getLogger` bezpośrednio. Konfiguracja raz na starcie:

```python
import logging
from polyneat import LoggingConfig, set_logging_config

set_logging_config(LoggingConfig(
    log_level=logging.DEBUG,
    file_log_directory="runs/logs",  # None = bez logów do pliku
))
```

### Własna funkcja fitness

```python
from polyneat.evaluators.sequential_evaluator_base import SequentialFitnessEvaluator
from polyneat.core.component_protocols import Phenotype
from polyneat.core.type_aliases import FitnessValue

class MojaOcena(SequentialFitnessEvaluator):
    def evaluate_single_phenotype(self, phenotype: Phenotype) -> FitnessValue:
        import torch
        wyniki = phenotype.forward_pass(torch.tensor([[1.0, 0.0]]))
        return float(wyniki[0, 0].item())
```

### Pełny eksperyment

```python
from pathlib import Path
import polyneat as pn
from polyneat.evaluators.xor_evaluator import XORFitnessEvaluator

config = pn.NEATConfig.load_from_yaml_file(Path("examples/xor_baseline.yaml"))
algorithm = pn.NEATAlgorithm.from_config(config)

runner = pn.EvolutionRunner(
    algorithm=algorithm,
    fitness_evaluator=XORFitnessEvaluator(),
    termination_criterion=pn.CompositeTermination([
        pn.TargetFitnessTermination(target_fitness=3.95),
        pn.MaxGenerationsTermination(max_generations=300),
    ]),
    callbacks=[
        pn.ConsoleStatisticsLogger(),
        pn.BestGenomePersister(output_directory=Path("runs")),
        pn.NetworkTopologyVisualizer(output_directory=Path("runs")),
        pn.TensorBoardLogger(log_directory=Path("runs"), run_name="xor"),
    ],
    random_seed=config.random_seed,
)

result = runner.run_evolution()
print(f"Najlepsza fitness: {result.best_fitness_ever_achieved:.4f}")
print(f"Powód zakończenia: {result.termination_reason}")
```

TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## Development

```bash
uv pip install -e ".[dev]"
ruff check polyneat        # linting
ruff format polyneat       # formatowanie
```

---

## Literatura

- Stanley, K. O. & Miikkulainen, R. (2002). **Evolving Neural Networks through Augmenting Topologies**. *Evolutionary Computation*, 10(2), 99–127.
- Pełny dokument projektowy: [`docs/superpowers/specs/2026-06-30-poly-neat-library-design.md`](docs/superpowers/specs/2026-06-30-poly-neat-library-design.md)
