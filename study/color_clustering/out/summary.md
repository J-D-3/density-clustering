# Color-clustering study — aggregated results

Runs: 392 total, 392 ok, 0 failed. max_dim=256.


## H2/H4 — Recall of expected colors (mean), by algorithm × space × kind

| config | rgb·orig | rgb·prep | lab·orig | lab·prep | overall |
|---|---|---|---|---|---|
| optics-thr | 0.55 | 0.63 | 0.46 | 0.67 | 0.58 |
| optics-xi05 | 0.83 | 0.63 | 0.85 | 0.66 | 0.74 |
| optics-xi01 | 0.29 | 0.60 | 0.77 | 0.61 | 0.57 |
| hdbscan-sm | 0.90 | 0.63 | 0.90 | 0.63 | 0.76 |
| hdbscan-lg | 0.89 | 0.69 | 0.90 | 0.69 | 0.79 |
| shdbscan | 0.92 | 0.63 | 0.92 | 0.63 | 0.77 |
| soptics | 0.38 | 0.56 | 0.30 | 0.58 | 0.46 |


_Space effect (Lab − RGB, mean recall over all ok runs):_ **0.033**


## H1 — Dominant cluster color & size, original vs preprocessed (hdbscan-lg)

| article | kind | top_color (mode) | mean top_frac | black-dominant? |
|---|---|---|---|---|
| amlodipin_10mg | original | white | 0.34 | 0/4 |
| amlodipin_10mg | preprocessed | white | 0.12 | 0/4 |
| amlodipin_5mg | original | white | 0.32 | 2/6 |
| amlodipin_5mg | preprocessed | white | 0.14 | 0/6 |
| bild_koeln | original | gray | 0.31 | 0/4 |
| bild_koeln | preprocessed | gray | 0.40 | 0/4 |
| spiegel_chatgpt | original | red | 0.46 | 0/6 |
| spiegel_chatgpt | preprocessed | brown | 0.63 | 0/6 |
| spiegel_hoffnung | original | blue | 0.24 | 0/4 |
| spiegel_hoffnung | preprocessed | blue | 0.31 | 0/4 |
| spiegel_mutante | original | black | 0.25 | 4/4 |
| spiegel_mutante | preprocessed | black | 0.24 | 3/4 |


## H3 — Preprocessing effect (mean over all ok runs of an image kind)

| metric | original | preprocessed |
|---|---|---|
| dedup collapse (×) | 3.7 | 10.9 |
| OPTICS/MST core (ms) | 2624 | 509 |
| n_clusters | 38.0 | 42.9 |
| recall | 0.705 | 0.631 |
| noise frac | 0.283 | 0.226 |


## Timing — median core (ms) and wall (s) by algorithm

| config | median core ms | median wall s | max wall s |
|---|---|---|---|
| optics-thr | 65 | 0.53 | 1.1 |
| optics-xi05 | 62 | 0.53 | 1.1 |
| optics-xi01 | 65 | 0.54 | 1.1 |
| hdbscan-sm | 173 | 0.60 | 2.2 |
| hdbscan-lg | 233 | 0.65 | 2.3 |
| shdbscan | 4613 | 5.03 | 32.9 |
| soptics | 700 | 1.19 | 3.8 |


## H6 — Approximate vs exact agreement (mean ARI / NMI)

| pair | kind | mean ARI | mean NMI | n |
|---|---|---|---|---|
| shdbscan vs hdbscan-sm | original | 0.552 | 0.633 | 28 |
| shdbscan vs hdbscan-sm | preprocessed | 0.816 | 0.860 | 28 |
| soptics vs optics-thr | original | 0.276 | 0.310 | 28 |
| soptics vs optics-thr | preprocessed | 0.754 | 0.775 | 28 |


## H5 — Same-article drift (cluster-count spread & mean matched ΔE across an article's images)

| article | kind | space | config | n_imgs | counts | Δcount | mean ΔE |
|---|---|---|---|---|---|---|---|
| amlodipin_10mg | original | lab | hdbscan-lg | 2 | [6, 8] | 2 | 12.6 |
| amlodipin_10mg | original | lab | optics-thr | 2 | [7, 5] | 2 | 3.7 |
| amlodipin_10mg | original | rgb | hdbscan-lg | 2 | [4, 6] | 2 | 3.5 |
| amlodipin_10mg | original | rgb | optics-thr | 2 | [7, 7] | 0 | 7.2 |
| amlodipin_10mg | preprocessed | lab | hdbscan-lg | 2 | [7, 6] | 1 | 3.0 |
| amlodipin_10mg | preprocessed | lab | optics-thr | 2 | [8, 9] | 1 | 5.2 |
| amlodipin_10mg | preprocessed | rgb | hdbscan-lg | 2 | [6, 10] | 4 | 2.4 |
| amlodipin_10mg | preprocessed | rgb | optics-thr | 2 | [10, 12] | 2 | 2.4 |
| amlodipin_5mg | original | lab | hdbscan-lg | 3 | [5, 15, 16] | 11 | 3.7 |
| amlodipin_5mg | original | lab | optics-thr | 3 | [3, 6, 7] | 4 | 5.7 |
| amlodipin_5mg | original | rgb | hdbscan-lg | 3 | [5, 7, 6] | 2 | 2.9 |
| amlodipin_5mg | original | rgb | optics-thr | 3 | [4, 8, 7] | 4 | 8.9 |
| amlodipin_5mg | preprocessed | lab | hdbscan-lg | 3 | [6, 7, 7] | 1 | 4.4 |
| amlodipin_5mg | preprocessed | lab | optics-thr | 3 | [7, 16, 7] | 9 | 2.6 |
| amlodipin_5mg | preprocessed | rgb | hdbscan-lg | 3 | [6, 7, 7] | 1 | 4.2 |
| amlodipin_5mg | preprocessed | rgb | optics-thr | 3 | [7, 16, 7] | 9 | 3.3 |
| bild_koeln | original | lab | hdbscan-lg | 2 | [4, 7] | 3 | 3.5 |
| bild_koeln | original | lab | optics-thr | 2 | [4, 2] | 2 | 17.6 |
| bild_koeln | original | rgb | hdbscan-lg | 2 | [5, 6] | 1 | 3.3 |
| bild_koeln | original | rgb | optics-thr | 2 | [3, 2] | 1 | 6.1 |
| bild_koeln | preprocessed | lab | hdbscan-lg | 2 | [8, 8] | 0 | 8.2 |
| bild_koeln | preprocessed | lab | optics-thr | 2 | [9, 10] | 1 | 9.0 |
| bild_koeln | preprocessed | rgb | hdbscan-lg | 2 | [7, 8] | 1 | 5.9 |
| bild_koeln | preprocessed | rgb | optics-thr | 2 | [8, 9] | 1 | 6.3 |
| spiegel_chatgpt | original | lab | hdbscan-lg | 3 | [4, 3, 5] | 2 | 5.0 |
| spiegel_chatgpt | original | lab | optics-thr | 3 | [3, 1, 2] | 2 | 7.2 |
| spiegel_chatgpt | original | rgb | hdbscan-lg | 3 | [4, 4, 4] | 0 | 3.3 |
| spiegel_chatgpt | original | rgb | optics-thr | 3 | [3, 3, 2] | 1 | 6.0 |
| spiegel_chatgpt | preprocessed | lab | hdbscan-lg | 3 | [5, 5, 4] | 1 | 4.1 |
| spiegel_chatgpt | preprocessed | lab | optics-thr | 3 | [7, 6, 6] | 1 | 9.5 |
| spiegel_chatgpt | preprocessed | rgb | hdbscan-lg | 3 | [4, 4, 4] | 0 | 4.1 |
| spiegel_chatgpt | preprocessed | rgb | optics-thr | 3 | [7, 9, 5] | 4 | 9.5 |
| spiegel_hoffnung | original | lab | hdbscan-lg | 2 | [6, 6] | 0 | 4.1 |
| spiegel_hoffnung | original | lab | optics-thr | 2 | [3, 4] | 1 | 15.6 |
| spiegel_hoffnung | original | rgb | hdbscan-lg | 2 | [7, 7] | 0 | 4.5 |
| spiegel_hoffnung | original | rgb | optics-thr | 2 | [5, 3] | 2 | 9.4 |
| spiegel_hoffnung | preprocessed | lab | hdbscan-lg | 2 | [9, 7] | 2 | 17.6 |
| spiegel_hoffnung | preprocessed | lab | optics-thr | 2 | [5, 10] | 5 | 4.1 |
| spiegel_hoffnung | preprocessed | rgb | hdbscan-lg | 2 | [5, 9] | 4 | 4.9 |
| spiegel_hoffnung | preprocessed | rgb | optics-thr | 2 | [6, 12] | 6 | 4.9 |
| spiegel_mutante | original | lab | hdbscan-lg | 2 | [7, 7] | 0 | 2.8 |
| spiegel_mutante | original | lab | optics-thr | 2 | [3, 2] | 1 | 4.9 |
| spiegel_mutante | original | rgb | hdbscan-lg | 2 | [8, 7] | 1 | 2.9 |
| spiegel_mutante | original | rgb | optics-thr | 2 | [3, 2] | 1 | 3.9 |
| spiegel_mutante | preprocessed | lab | hdbscan-lg | 2 | [8, 6] | 2 | 3.2 |
| spiegel_mutante | preprocessed | lab | optics-thr | 2 | [6, 7] | 1 | 8.2 |
| spiegel_mutante | preprocessed | rgb | hdbscan-lg | 2 | [9, 7] | 2 | 6.3 |
| spiegel_mutante | preprocessed | rgb | optics-thr | 2 | [6, 6] | 0 | 5.6 |


## Voxel sub-study — collapse / cluster count / recall / core-time vs bin

| article | space | algo | bin | collapse | k | recall | core ms |
|---|---|---|---|---|---|---|---|
| bild_koeln | rgb | optics-thr | 0 | 3.2 | 3 | 0.25 | 158 |
| bild_koeln | rgb | optics-thr | 4 | 11.6 | 4 | 0.50 | 19 |
| bild_koeln | rgb | optics-thr | 8 | 51.6 | 43 | 0.75 | 3 |
| bild_koeln | rgb | optics-thr | 16 | 240.0 | 52 | 0.75 | 0 |
| bild_koeln | rgb | hdbscan-sm | 0 | 3.2 | 9 | 0.75 | 834 |
| bild_koeln | rgb | hdbscan-sm | 4 | 11.6 | 8 | 0.75 | 58 |
| bild_koeln | rgb | hdbscan-sm | 8 | 51.6 | 1 | 0.00 | 7 |
| bild_koeln | rgb | hdbscan-sm | 16 | 240.0 | 1 | 0.00 | 4 |
| bild_koeln | lab | optics-thr | 0 | 3.2 | 4 | 0.50 | 167 |
| bild_koeln | lab | optics-thr | 2 | 12.8 | 4 | 0.50 | 17 |
| bild_koeln | lab | optics-thr | 4 | 58.7 | 47 | 0.75 | 2 |
| bild_koeln | lab | optics-thr | 8 | 288.6 | 50 | 0.75 | 0 |
| bild_koeln | lab | hdbscan-sm | 0 | 3.2 | 23 | 0.75 | 703 |
| bild_koeln | lab | hdbscan-sm | 2 | 12.8 | 7 | 0.75 | 42 |
| bild_koeln | lab | hdbscan-sm | 4 | 58.7 | 1 | 0.00 | 6 |
| bild_koeln | lab | hdbscan-sm | 8 | 288.6 | 1 | 0.00 | 3 |
| spiegel_mutante | rgb | optics-thr | 0 | 4.1 | 3 | 0.50 | 166 |
| spiegel_mutante | rgb | optics-thr | 4 | 11.5 | 2 | 0.50 | 30 |
| spiegel_mutante | rgb | optics-thr | 8 | 34.0 | 2 | 0.50 | 6 |
| spiegel_mutante | rgb | optics-thr | 16 | 132.9 | 36 | 1.00 | 1 |
| spiegel_mutante | rgb | hdbscan-sm | 0 | 4.1 | 8 | 1.00 | 484 |
| spiegel_mutante | rgb | hdbscan-sm | 4 | 11.5 | 7 | 1.00 | 65 |
| spiegel_mutante | rgb | hdbscan-sm | 8 | 34.0 | 4 | 0.75 | 12 |
| spiegel_mutante | rgb | hdbscan-sm | 16 | 132.9 | 2 | 0.25 | 4 |
| spiegel_mutante | lab | optics-thr | 0 | 4.1 | 3 | 0.50 | 212 |
| spiegel_mutante | lab | optics-thr | 2 | 13.6 | 1 | 0.25 | 24 |
| spiegel_mutante | lab | optics-thr | 4 | 41.6 | 1 | 0.25 | 4 |
| spiegel_mutante | lab | optics-thr | 8 | 169.3 | 27 | 1.00 | 1 |
| spiegel_mutante | lab | hdbscan-sm | 0 | 4.1 | 13 | 1.00 | 418 |
| spiegel_mutante | lab | hdbscan-sm | 2 | 13.6 | 5 | 1.00 | 42 |
| spiegel_mutante | lab | hdbscan-sm | 4 | 41.6 | 3 | 0.75 | 9 |
| spiegel_mutante | lab | hdbscan-sm | 8 | 169.3 | 1 | 0.00 | 4 |
