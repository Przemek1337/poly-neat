# Uruchomienie benchmarku na serwerze Linux / NVIDIA

Komendy wykonuj z katalogu repozytorium. Jeden proces badawczy na jednej
wyłącznej karcie GPU. Najpierw smoke, potem pilot, następnie zamrożenie
protokołu i seria full. Pilot nie wczytuje tensorów threshold_validation ani
official_test i nie zapisuje ich metryk. Audyt może czytać pliki testowe
wyłącznie do kontroli integralności.

## 1. Kod i środowisko

W istniejącym checkoutcie, po sprawdzeniu czy nie ma lokalnych zmian:

```bash
git status --short
git switch feat/deepneat-implementation
git pull --ff-only
uv sync --locked --extra benchmark
nvidia-smi
```

Nie usuwaj lokalnych zmian w razie konfliktu. `uv sync --locked` ma korzystać
z wersji w repo, a nie dobierać nowych wersji bibliotek na potrzeby serii.
Sprawdź CUDA i zgodność torchvision z PyTorch:

```bash
uv run --no-sync python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); b=torch.tensor([[0.,0.,1.,1.]],device='cuda'); print(torchvision.ops.nms(b,torch.ones(1,device='cuda'),0.5))"
```

Jeśli ta kontrola zawiedzie, zatrzymaj się: CPU fallback nie jest właściwym
rozwiązaniem dla budżetu zaplanowanego na GPU. Nie zmieniaj ręcznie wersji
torch podczas serii. Zgodność konkretnego sterownika i instalacji musi zostać
potwierdzona na serwerze; testy lokalne nie zastępują tej kontroli.

## 2. Testy i smoke

```bash
uv run --no-sync pytest -q tests/test_pneumonia_execution.py tests/test_resumable_search.py tests/test_pediatric_pneumonia_full_profiles.py tests/test_device_placement.py tests/test_benchmark_runner.py

uv run --no-sync python -m examples.pediatric_pneumonia.deepneat --gpu --mode smoke --artifacts-directory benchmarks/results/pneumonia/smoke/deepneat
uv run --no-sync python -m examples.pediatric_pneumonia.exact --gpu --mode smoke --artifacts-directory benchmarks/results/pneumonia/smoke/exact
uv run --no-sync python -m examples.pediatric_pneumonia.random_search --gpu --mode smoke --artifacts-directory benchmarks/results/pneumonia/smoke/random_search
uv run --no-sync python -m examples.pediatric_pneumonia.fixed_cnn --gpu --mode smoke --artifacts-directory benchmarks/results/pneumonia/smoke/fixed_cnn
```

Domyślne YAML-e tych modułów to smoke. Używają syntetycznych obrazów i ich
wyniki nie nadają się do tabel w pracy. Kolejne świeże wykonanie wymaga nowego
katalogu; `--resume` służy kontynuacji tego samego przebiegu.

Transfer learning jest opcjonalnym, osobno raportowanym baseline'em:

```bash
uv run --no-sync python -m examples.pediatric_pneumonia.transfer_learning --gpu --mode smoke --artifacts-directory benchmarks/results/pneumonia/smoke/transfer_learning
```

Pierwsze wykonanie może pobrać wagi ImageNet. Cztery główne metody nie
wymagają dostępu do tych wag.

## 3. Dataset

Po skonfigurowaniu uwierzytelnienia Kaggle na serwerze:

```bash
mkdir -p examples/pediatric_pneumonia/data/download
uvx --from kaggle kaggle datasets download -d andrewmvd/pediatric-pneumonia-chest-xray -p examples/pediatric_pneumonia/data/download
unzip examples/pediatric_pneumonia/data/download/pediatric-pneumonia-chest-xray.zip -d examples/pediatric_pneumonia/data/kaggle
```

Zachowaj archiwum i metadane jego wersji/licencji. Dalsze komendy zakładają,
że loader rozpozna drzewo pod `examples/pediatric_pneumonia/data/kaggle`.
Obsługiwane są opisane w README poziomy zagnieżdżenia `chest_xray`.
Nie przenoś obrazów między splitami, aby ominąć błąd audytu.

## 4. Pilot na rzeczywistych danych

Uruchamiaj kolejno, np. w sesji `tmux`, aby rozłączenie SSH nie kończyło pracy:

```bash
tmux new -s pneumonia
```

W sesji:

```bash
for method in deepneat exact random_search fixed_cnn; do
  uv run --no-sync python -m examples.pediatric_pneumonia.$method \
    --gpu --mode pilot --seed 7 \
    --config examples/pediatric_pneumonia/configs/${method}_full.yaml \
    --data-directory examples/pediatric_pneumonia/data/kaggle \
    --artifacts-directory benchmarks/results/pneumonia/pilot/$method || break
done
```

Odłączenie od tmux: Ctrl-b, następnie d. Powrót: `tmux attach -t pneumonia`.
Ustawienie `CUDA_VISIBLE_DEVICES` przed procesem pozwala wybrać przydzieloną
kartę; nie uruchamiaj kilku metod równocześnie na tej samej karcie.

Pliki nazwane `_full.yaml` są kandydatami konfiguracji, nie zamrożonym
protokołem. Domyślny limit wyszukiwania w głównych profilach wynosi 7200 s,
więc pilot nie jest kilkusekundowym testem. Stały CNN ma osobny budżet epok.
Generacje/liczba losowań są dodatkowymi limitami, więc wyszukiwanie może
zakończyć się wcześniej; odnotuj to przy interpretacji wyników.

Sprawdź `run_report.json` i logi: AUROC search_validation, liczbę ocen,
odsetek błędów, czasy oraz komunikaty audytu. Pole `official_test` musi być
puste. Jeśli zmieniasz budżety lub recepty, przeprowadź pilot nowych profili
w nowych katalogach. Możesz pracować na kopii katalogu konfiguracji i użyć
tej samej kopii przy `--profiles-directory` w następnym kroku. Dla wspólnego
toru B zachowaj identyczną receptę w czterech profilach.

## 5. Zamrożenie serii po zaakceptowaniu pilotażu

Poniższe dwa placeholdery trzeba zastąpić zweryfikowaną wersją i licencją
rzeczywiście pobranego datasetu. Nie wpisuj ich na podstawie samej nazwy zbioru.
Wybierz nowy identyfikator badania i nowy katalog wyjściowy.

```bash
uv run --no-sync python -m examples.pediatric_pneumonia.freeze \
  --gpu \
  --data-directory examples/pediatric_pneumonia/data/kaggle \
  --output-directory benchmarks/results/pneumonia/series-v1 \
  --protocol-id pediatric-pneumonia-series-v1 \
  --dataset-release 'WPISZ_ZWERYFIKOWANA_WERSJE_KAGGLE' \
  --dataset-license 'WPISZ_ZWERYFIKOWANA_LICENCJE' \
  --seeds 101 102 103 104 105 \
  --pilot-reports \
    benchmarks/results/pneumonia/pilot/deepneat/run_report.json \
    benchmarks/results/pneumonia/pilot/exact/run_report.json \
    benchmarks/results/pneumonia/pilot/random_search/run_report.json \
    benchmarks/results/pneumonia/pilot/fixed_cnn/run_report.json
```

Powstaną `protocol.lock.yaml` w schemacie 2.0, manifest, kopie profili
`profiles/<method>.yaml` i raporty pilotażu. Komenda sprawdza, czy każda metoda
ma udany pilot na wskazanej konfiguracji, tym samym archiwum i środowisku,
czy seedy pilota i wyników są rozłączne oraz czy wspólne recepty są zgodne.
Nie wykonuje treningu, wyboru progu ani testu. Nie nadpisuje istniejącej serii.

Stary `configs/protocol.lock.template.yaml` w schemacie 1.0 pozostaje materiałem
opisowym. Nie jest akceptowany przez nowe wykonanie full. Nie edytuj ręcznie
wygenerowanego locka ani zamrożonych profili; zmiany oznaczają nowe badanie.

## 6. Seria wynikowa

```bash
for method in deepneat exact random_search fixed_cnn; do
  uv run --no-sync python -m benchmarks.run_benchmark pediatric_pneumonia/$method \
    --gpu --mode full \
    --config benchmarks/results/pneumonia/series-v1/profiles/${method}.yaml \
    --protocol-lock benchmarks/results/pneumonia/series-v1/protocol.lock.yaml \
    --data-directory examples/pediatric_pneumonia/data/kaggle \
    --repeats 5 --base-seed 101 \
    --artifacts-root benchmarks/results/pneumonia/series-v1/runs || break
done
```

Full odrzuca zmieniony lock, profil, kod/środowisko, niezadeklarowany seed,
inne archiwum/manifest i blokujące wyniki audytu. Dopiero ten tryb wykonuje
końcowe A/B, wybór progów i test na rzeczywistych danych.
Nie dostrajaj kolejnych metod po zobaczeniu wyników testu.

## 7. Przerwanie i wznowienie

Wznowienie pilota, np. EXACT:

```bash
uv run --no-sync python -m examples.pediatric_pneumonia.exact \
  --gpu --mode pilot --seed 7 --resume \
  --config examples/pediatric_pneumonia/configs/exact_full.yaml \
  --data-directory examples/pediatric_pneumonia/data/kaggle \
  --artifacts-directory benchmarks/results/pneumonia/pilot/exact
```

Aby wznowić serię, dodaj `--resume` do tej samej komendy `benchmarks.run_benchmark`.
Ukończone raporty są odczytywane po kontroli zgodności, bez ponownego testowania;
istniejące wyszukiwanie jest kontynuowane. Nowe seedy bez checkpointu uruchamiaj
bez `--resume`, np. zmieniając `--base-seed` i `--repeats` na pozostały zakres.

Zapis obejmuje ukończone oceny, populację, specjację, innowacje, stan scorerów
i RNG oraz wytrenowane wagi zwycięzcy. Niedokończona ocena lub reprodukcja są
powtarzane od ostatniej granicy. Czas utraconej pracy pozostaje w budżecie.
Wznowienie dotyczy wyszukiwania; przerwany retrening B może zostać wykonany
ponownie po odtworzeniu wybranego modelu. Nie deklarujemy wznowienia od środka
batcha/epoki B.

Ctrl-C zapisuje rozliczenie czasu. Po `kill -9`, awarii zasilania lub
przymusowym zakończeniu przez scheduler nie da się automatycznie ustalić
całego utraconego czasu: wznowienie wymaga dodatkowo
`--lost-work-seconds LICZBA_SEKUND`. Jest to czas aktywnej pracy od ostatniego
rekordu `search/budget.json` do awarii, ustalony z logów/schedulera,
bez czasu postoju. Nie wpisuj zera bez potwierdzenia.

Nie uruchamiaj równocześnie dwóch procesów w tym samym katalogu artefaktów.
Nie czyść `search/`, aby uzyskać ponownie pełny budżet tej samej serii.

## 8. Gdzie są wyniki

Przykładowy seed:

```text
benchmarks/results/pneumonia/series-v1/runs/pediatric_pneumonia_deepneat/seed_101/
  manifest.json
  run_report.json
  search/latest.json
  search/budget.json
  search/state_*.pt
  checkpoints/*.pt
  predictions/*.json
```

Nieudane przebiegi i ograniczenia danych też należą do raportu. Przejście
testów automatycznych nie gwarantuje jakości modelu ani braku problemów
z rzeczywistym datasetem. Plany w docs i wygenerowane artefakty nie są częścią
commita przygotowującego ten benchmark.
