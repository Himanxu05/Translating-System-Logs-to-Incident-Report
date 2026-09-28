
import os
import re
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Attention, Concatenate,
    Bidirectional, Dropout, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_synthetic_dataset(num_samples: int = 500) -> list[dict]:
    templates = [

        {
            "log_pattern": (
                "[{ts}] INFO  kernel: CPU usage {cpu}% on core {core} | "
                "[{ts}] WARN  scheduler: process '{proc}' consuming {cpu}% CPU | "
                "[{ts}] ERROR watchdog: CPU threshold exceeded on host {host} | "
                "[{ts}] CRIT  alertmanager: sustained high CPU for {dur} seconds"
            ),
            "report_template": (
                "INCIDENT REPORT: CPU Saturation on {host}. "
                "Root Cause: Process '{proc}' consumed {cpu}% CPU on core {core} for {dur} seconds, "
                "exceeding the 80% alerting threshold. "
                "Impact: Elevated response latency and potential request timeouts on affected host. "
                "Recommended Action: Investigate '{proc}' for infinite loops or resource leaks; "
                "consider cgroups CPU limits."
            )
        },

        {
            "log_pattern": (
                "[{ts}] INFO  meminfo: total={mem_total}MB free={mem_free}MB | "
                "[{ts}] WARN  jvm: heap usage at {heap}% on service {svc} | "
                "[{ts}] ERROR oom_killer: out-of-memory on host {host}, killing {proc} | "
                "[{ts}] CRIT  monit: service {svc} restarted after OOM event"
            ),
            "report_template": (
                "INCIDENT REPORT: Memory Exhaustion on {host}. "
                "Root Cause: Service '{svc}' experienced a memory leak with heap utilization "
                "reaching {heap}%, triggering the OOM killer to terminate '{proc}'. "
                "Impact: Service '{svc}' was forcefully restarted, causing a brief outage. "
                "Recommended Action: Analyze heap dumps for '{svc}'; review recent code changes "
                "for unbounded caching or unclosed connections."
            )
        },

        {
            "log_pattern": (
                "[{ts}] INFO  iostat: disk {disk} util={util}% r={r}MB/s w={w}MB/s | "
                "[{ts}] WARN  kernel: I/O wait time {iowait}% on {host} | "
                "[{ts}] ERROR db: query timeout after {dur}ms due to slow disk | "
                "[{ts}] CRIT  alertmanager: disk {disk} I/O saturation on {host}"
            ),
            "report_template": (
                "INCIDENT REPORT: Disk I/O Saturation on {host}. "
                "Root Cause: Disk '{disk}' reached {util}% utilization with I/O wait at {iowait}%, "
                "causing database queries to timeout after {dur}ms. "
                "Impact: Database read/write operations degraded, affecting dependent services. "
                "Recommended Action: Identify top I/O consumers via iotop; "
                "evaluate SSD upgrade or read replica offloading."
            )
        },

        {
            "log_pattern": (
                "[{ts}] INFO  netstat: connections={conns} on interface {iface} | "
                "[{ts}] WARN  app: upstream timeout connecting to {remote} | "
                "[{ts}] ERROR nginx: 504 Gateway Timeout to {remote} after {dur}ms | "
                "[{ts}] CRIT  alertmanager: network latency {lat}ms to {remote} from {host}"
            ),
            "report_template": (
                "INCIDENT REPORT: Network Timeout to {remote} from {host}. "
                "Root Cause: Upstream service '{remote}' failed to respond within {dur}ms, "
                "resulting in 504 Gateway Timeout errors. Network latency measured at {lat}ms. "
                "Impact: End-user requests to endpoints dependent on '{remote}' returned errors. "
                "Recommended Action: Verify '{remote}' health; check firewall rules "
                "and increase circuit-breaker timeout threshold."
            )
        },

        {
            "log_pattern": (
                "[{ts}] INFO  systemd: starting {svc}.service | "
                "[{ts}] ERROR {svc}: unhandled exception — {error} | "
                "[{ts}] ERROR systemd: {svc}.service failed with exit code {code} | "
                "[{ts}] CRIT  pagerduty: {svc} down on {host}, {failures} failures in {dur}s"
            ),
            "report_template": (
                "INCIDENT REPORT: Service Crash — {svc} on {host}. "
                "Root Cause: Service '{svc}' terminated with exit code {code} due to "
                "unhandled exception '{error}'. {failures} failures detected within {dur} seconds. "
                "Impact: Complete service unavailability for '{svc}'; dependent services degraded. "
                "Recommended Action: Review application logs for '{error}'; "
                "deploy hotfix and enable automatic restart with backoff."
            )
        },

        {
            "log_pattern": (
                "[{ts}] INFO  app: connecting to DB at {db_host}:{db_port} | "
                "[{ts}] WARN  pool: connection pool at {pool}% capacity | "
                "[{ts}] ERROR app: FATAL connection refused to {db_host}:{db_port} | "
                "[{ts}] CRIT  alertmanager: DB unreachable from {host} for {dur}s"
            ),
            "report_template": (
                "INCIDENT REPORT: Database Connection Failure on {host}. "
                "Root Cause: Application on '{host}' could not establish connection to "
                "database at '{db_host}:{db_port}'. Connection pool was at {pool}% capacity "
                "before full failure; DB unreachable for {dur} seconds. "
                "Impact: All database-backed operations failed; application returned 500 errors. "
                "Recommended Action: Verify DB instance health; check max_connections setting "
                "and implement connection pooler (PgBouncer/ProxySQL)."
            )
        },
    ]

    dataset = []
    base_ts = "2024-03-15T"

    for _ in range(num_samples):
        t = templates[random.randint(0, len(templates) - 1)]
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        ts = f"{base_ts}{hour:02d}:{minute:02d}:00Z"

        vals = {
            "ts": ts,
            "cpu": random.randint(82, 99),
            "core": random.randint(0, 15),
            "proc": random.choice(["java", "python3", "node", "mysqld", "nginx", "redis-server"]),
            "host": f"prod-{random.choice(['web', 'db', 'cache', 'api'])}-{random.randint(1, 8):02d}",
            "dur": random.randint(30, 600),
            "mem_total": 32768,
            "mem_free": random.randint(100, 800),
            "heap": random.randint(88, 99),
            "svc": random.choice(["auth-service", "payment-api", "order-processor",
                                   "user-service", "notification-svc"]),
            "disk": random.choice(["sda", "nvme0n1", "sdb"]),
            "util": random.randint(91, 100),
            "r": random.randint(200, 600),
            "w": random.randint(100, 400),
            "iowait": random.randint(40, 90),
            "conns": random.randint(500, 5000),
            "iface": random.choice(["eth0", "ens3", "bond0"]),
            "remote": random.choice(["payment-gateway.prod", "auth.internal", "db-primary:5432"]),
            "lat": random.randint(800, 9000),
            "error": random.choice([
                "NullPointerException", "SegmentationFault", "RuntimeError: out of resources",
                "ConnectionResetError", "KeyError: 'user_id'"
            ]),
            "code": random.choice([1, 2, 137, 139]),
            "failures": random.randint(3, 20),
            "db_host": random.choice(["db-primary", "pg-master", "mysql-01"]),
            "db_port": random.choice([5432, 3306, 27017]),
            "pool": random.randint(92, 100),
        }

        log_text = t["log_pattern"].format(**vals)
        report_text = t["report_template"].format(**vals)

        dataset.append({
            "log": log_text,
            "report": "<start> " + report_text + " <end>"
        })

    return dataset


class LogPreprocessor:
    def __init__(self, num_words: int = 5000, max_log_len: int = 120,
                 max_report_len: int = 80):
        self.num_words = num_words
        self.max_log_len = max_log_len
        self.max_report_len = max_report_len
        self.log_tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>",
                                       filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t')
        self.report_tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>",
                                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t')

    def normalize_log(self, log: str) -> str:
        log = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '<TIMESTAMP>', log)
        log = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', log)
        log = re.sub(r'\b\d+\b', '<NUM>', log)
        return log.lower()

    def fit(self, logs: list[str], reports: list[str]):
        norm_logs = [self.normalize_log(l) for l in logs]
        self.log_tokenizer.fit_on_texts(norm_logs)
        self.report_tokenizer.fit_on_texts(reports)
        self.log_vocab_size = min(self.num_words, len(self.log_tokenizer.word_index) + 1)
        self.report_vocab_size = min(self.num_words, len(self.report_tokenizer.word_index) + 1)
        self.start_token = self.report_tokenizer.word_index.get("<start>", 1)
        self.end_token = self.report_tokenizer.word_index.get("<end>", 2)

    def transform(self, logs: list[str], reports: list[str]):
        norm_logs = [self.normalize_log(l) for l in logs]
        X = self.log_tokenizer.texts_to_sequences(norm_logs)
        X = pad_sequences(X, maxlen=self.max_log_len, padding="post", truncating="post")
        Y = self.report_tokenizer.texts_to_sequences(reports)
        Y = pad_sequences(Y, maxlen=self.max_report_len, padding="post", truncating="post")
        return X.astype(np.int32), Y.astype(np.int32)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)

    def call(self, query, values):
        query_expanded = tf.expand_dims(query, 1)  
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)       
        return context, attention_weights


def build_seq2seq_model(log_vocab_size: int, report_vocab_size: int,
                         embed_dim: int = 128, lstm_units: int = 256,
                         max_log_len: int = 120, dropout: float = 0.3):
    enc_input = Input(shape=(max_log_len,), name="encoder_input")
    enc_embed = Embedding(log_vocab_size, embed_dim, mask_zero=True, name="enc_embed")(enc_input)
    enc_embed = Dropout(dropout)(enc_embed)
    enc_out, fwd_h, fwd_c, bwd_h, bwd_c = Bidirectional(
        LSTM(lstm_units, return_sequences=True, return_state=True,
             recurrent_dropout=0.1), name="enc_bilstm"
    )(enc_embed)
    enc_state_h = Concatenate()([fwd_h, bwd_h])           
    enc_state_c = Concatenate()([fwd_c, bwd_c])


    state_h = Dense(lstm_units, activation="tanh", name="state_h_proj")(enc_state_h)
    state_c = Dense(lstm_units, activation="tanh", name="state_c_proj")(enc_state_c)

    dec_input = Input(shape=(None,), name="decoder_input")
    dec_embed = Embedding(report_vocab_size, embed_dim, mask_zero=True, name="dec_embed")(dec_input)
    dec_embed = Dropout(dropout)(dec_embed)
    dec_lstm  = LSTM(lstm_units, return_sequences=True, return_state=True,
                     recurrent_dropout=0.1, name="dec_lstm")
    dec_out, _, _ = dec_lstm(dec_embed, initial_state=[state_h, state_c])


    enc_proj = Dense(lstm_units, name="enc_proj")(enc_out)
    attention_layer = BahdanauAttention(lstm_units // 2)

    context_vectors = []
    for t in range(1): 
        pass

    attention = tf.keras.layers.Attention(name="attention")
    context = attention([dec_out, enc_proj])  

    dec_combined = Concatenate(axis=-1, name="dec_concat")([dec_out, context])
    dec_combined = LayerNormalization()(dec_combined)
    dec_combined = Dropout(dropout)(dec_combined)

    output = Dense(report_vocab_size, activation="softmax", name="output")(dec_combined)

    model = Model([enc_input, dec_input], output, name="IncidentReportGenerator")
    return model, state_h, state_c, enc_proj


def build_inference_components(model, log_vocab_size, report_vocab_size,
                                 embed_dim, lstm_units, max_log_len, dropout=0.3):
    
    enc_input = model.get_layer("encoder_input").input
    enc_out   = model.get_layer("enc_bilstm").output   
    sh        = model.get_layer("state_h_proj").output
    sc        = model.get_layer("state_c_proj").output
    enc_proj  = model.get_layer("enc_proj").output

    encoder_model = Model(enc_input, [enc_proj, sh, sc], name="encoder_inference")

    dec_state_h_in = Input(shape=(lstm_units,), name="dec_state_h_input")
    dec_state_c_in = Input(shape=(lstm_units,), name="dec_state_c_input")
    enc_proj_in    = Input(shape=(max_log_len, lstm_units), name="enc_proj_input")
    dec_tok_in     = Input(shape=(1,), name="dec_token_input")

    dec_embed_inf  = model.get_layer("dec_embed")(dec_tok_in)
    dec_out_inf, out_h, out_c = model.get_layer("dec_lstm")(
        dec_embed_inf, initial_state=[dec_state_h_in, dec_state_c_in]
    )
    context_inf = model.get_layer("attention")([dec_out_inf, enc_proj_in])
    combined_inf = tf.concat([dec_out_inf, context_inf], axis=-1)

   
    combined_inf = model.get_layer("dec_concat")([dec_out_inf, context_inf])
    dense_out    = model.get_layer("output")(combined_inf)

    decoder_model = Model(
        [dec_tok_in, dec_state_h_in, dec_state_c_in, enc_proj_in],
        [dense_out, out_h, out_c],
        name="decoder_inference"
    )
    return encoder_model, decoder_model


def prepare_decoder_inputs_targets(Y: np.ndarray):
    dec_in  = Y[:, :-1]
    dec_out = Y[:, 1:]
    return dec_in, dec_out


def train_model(model, X_train, dec_in_train, dec_out_train,
                X_val, dec_in_val, dec_out_val,
                report_vocab_size: int, epochs: int = 30, batch_size: int = 32):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=5.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        [X_train, dec_in_train],
        dec_out_train,
        validation_data=([X_val, dec_in_val], dec_out_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


def generate_report_greedy(log_text: str, model: Model,
                            preprocessor: LogPreprocessor,
                            max_len: int = 80) -> str:
    norm_log = preprocessor.normalize_log(log_text)
    seq = preprocessor.log_tokenizer.texts_to_sequences([norm_log])
    padded = pad_sequences(seq, maxlen=preprocessor.max_log_len,
                           padding="post", truncating="post")

    generated_ids = [preprocessor.start_token]

    for _ in range(max_len):
        dec_in = pad_sequences([generated_ids], maxlen=preprocessor.max_report_len,
                                padding="post")[:, :-1]
        preds = model.predict([padded, dec_in], verbose=0)       
        next_id = int(np.argmax(preds[0, len(generated_ids) - 1]))  
        if next_id == preprocessor.end_token or next_id == 0:
            break
        generated_ids.append(next_id)

    id2word = {v: k for k, v in preprocessor.report_tokenizer.word_index.items()}
    tokens = [id2word.get(i, "") for i in generated_ids[1:]]
    report = " ".join(t for t in tokens if t not in ("", "<OOV>", "<end>", "<start>"))
    return report.capitalize()



def simple_bleu_1gram(reference: str, hypothesis: str) -> float:
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    ref_set = set(ref_tokens)
    matches = sum(1 for t in hyp_tokens if t in ref_set)
    precision = matches / len(hyp_tokens)
    brevity   = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    return round(brevity * precision, 4)


def evaluate_model(model, test_logs, test_reports, preprocessor, n_samples=10):
    print("\n" + "═" * 70)
    print("  EVALUATION — Qualitative Sample Outputs & BLEU-1")
    print("═" * 70)
    bleu_scores = []
    for i in range(min(n_samples, len(test_logs))):
        pred = generate_report_greedy(test_logs[i], model, preprocessor)
        ref  = test_reports[i].replace("<start> ", "").replace(" <end>", "")
        b    = simple_bleu_1gram(ref, pred)
        bleu_scores.append(b)
        print(f"\n[Sample {i+1}]")
        print(f"  LOG   : {test_logs[i][:120]}...")
        print(f"  PRED  : {pred[:200]}")
        print(f"  REF   : {ref[:200]}")
        print(f"  BLEU-1: {b:.4f}")

    avg = np.mean(bleu_scores)
    print(f"\n  Average BLEU-1 over {len(bleu_scores)} samples: {avg:.4f}")
    print("═" * 70)
    return avg


def rule_based_baseline(log_text: str) -> str:
    
    log_lower = log_text.lower()
    if "cpu" in log_lower and ("threshold" in log_lower or "saturation" in log_lower):
        return "INCIDENT: High CPU usage detected. Impact: Performance degradation. Action: Check top processes."
    elif "oom" in log_lower or "out-of-memory" in log_lower:
        return "INCIDENT: Out-of-memory event. Impact: Service restart. Action: Analyze memory usage."
    elif "disk" in log_lower and ("util" in log_lower or "i/o" in log_lower):
        return "INCIDENT: Disk I/O issue detected. Impact: Slow queries. Action: Review I/O usage."
    elif "timeout" in log_lower or "gateway" in log_lower:
        return "INCIDENT: Network timeout detected. Impact: Request failures. Action: Check upstream service."
    elif "connection refused" in log_lower or "db" in log_lower:
        return "INCIDENT: Database connectivity issue. Impact: Application errors. Action: Verify DB health."
    elif "exit code" in log_lower or "failed" in log_lower:
        return "INCIDENT: Service failure detected. Impact: Service unavailable. Action: Review logs."
    else:
        return "INCIDENT: Anomaly detected in system logs. Action: Manual investigation required."

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("  Automated Incident Report Generator — TensorFlow Pipeline")

    dataset = create_synthetic_dataset(num_samples=500)
    random.shuffle(dataset)

    logs    = [d["log"]    for d in dataset]
    reports = [d["report"] for d in dataset]

    split = int(0.8 * len(dataset))
    train_logs, val_logs   = logs[:split], logs[split:]
    train_reps, val_reps   = reports[:split], reports[split:]

    print(f"    Train: {len(train_logs)} | Val: {len(val_logs)}")
    print(f"    Sample log    : {train_logs[0][:100]}...")
    print(f"    Sample report : {train_reps[0][:100]}...")


    prep = LogPreprocessor(num_words=3000, max_log_len=100, max_report_len=70)
    prep.fit(train_logs, train_reps)
    X_train, Y_train = prep.transform(train_logs, train_reps)
    X_val,   Y_val   = prep.transform(val_logs,   val_reps)

    dec_in_train, dec_out_train = prepare_decoder_inputs_targets(Y_train)
    dec_in_val,   dec_out_val   = prepare_decoder_inputs_targets(Y_val)

    print(f"    Log vocab size   : {prep.log_vocab_size}")
    print(f"    Report vocab size: {prep.report_vocab_size}")
    print(f"    X_train shape    : {X_train.shape}")
    print(f"    Y_train shape    : {Y_train.shape}")

    EMBED_DIM  = 128
    LSTM_UNITS = 256
    DROPOUT    = 0.3

    model, _, _, _ = build_seq2seq_model(
        log_vocab_size=prep.log_vocab_size,
        report_vocab_size=prep.report_vocab_size,
        embed_dim=EMBED_DIM,
        lstm_units=LSTM_UNITS,
        max_log_len=prep.max_log_len,
        dropout=DROPOUT
    )
    model.summary()

    history = train_model(
        model, X_train, dec_in_train, dec_out_train,
        X_val, dec_in_val, dec_out_val,
        report_vocab_size=prep.report_vocab_size,
        epochs=25,
        batch_size=32
    )

    best_val_loss = min(history.history["val_loss"])
    print(f"\n    Best validation loss: {best_val_loss:.4f}")


    model.save("incident_report_model.keras")
    print("    Model saved to: incident_report_model.keras")

    unseen_examples = [
        ("[2024-03-16T03:15:00Z] INFO meminfo: total=32768MB free=210MB | "
         "[2024-03-16T03:15:02Z] WARN jvm: heap usage at 97% on service payment-api | "
         "[2024-03-16T03:15:05Z] ERROR oom_killer: out-of-memory on host prod-api-03, killing java | "
         "[2024-03-16T03:15:06Z] CRIT monit: service payment-api restarted after OOM event"),

        ("[2024-03-16T14:22:00Z] INFO iostat: disk nvme0n1 util=98% r=350MB/s w=180MB/s | "
         "[2024-03-16T14:22:01Z] WARN kernel: I/O wait time 76% on prod-db-01 | "
         "[2024-03-16T14:22:03Z] ERROR db: query timeout after 12000ms due to slow disk | "
         "[2024-03-16T14:22:04Z] CRIT alertmanager: disk nvme0n1 I/O saturation on prod-db-01"),

        ("[2024-03-16T09:00:00Z] INFO app: connecting to DB at pg-master:5432 | "
         "[2024-03-16T09:00:01Z] WARN pool: connection pool at 99% capacity | "
         "[2024-03-16T09:00:03Z] ERROR app: FATAL connection refused to pg-master:5432 | "
         "[2024-03-16T09:00:05Z] CRIT alertmanager: DB unreachable from prod-web-02 for 120s"),
    ]

    for i, log in enumerate(unseen_examples):
        print(f"\n  [Unseen {i+1}]")
        print(f"  LOG  : {log[:120]}...")
        pred = generate_report_greedy(log, model, prep)
        print(f"  PRED : {pred}")

    print("\n  Rule-Based Baseline Comparison ")
    for i, log in enumerate(unseen_examples):
        print(f"\n  [Unseen {i+1}]")
        print(f"  RULE : {rule_based_baseline(log)}")

    avg_bleu = evaluate_model(model, val_logs, val_reps, prep, n_samples=5)


    print("\n  TRAINING SUMMARY")
    print(f"  Final Train Loss  : {history.history['loss'][-1]:.4f}")
    print(f"  Final Val Loss    : {history.history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc   : {history.history['accuracy'][-1]:.4f}")
    print(f"  Final Val Acc     : {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Avg BLEU-1 (val)  : {avg_bleu:.4f}")
    print(f"  Model Parameters  : {model.count_params():,}")
    print("\n  Pipeline complete. Model saved.\n")


if __name__ == "__main__":
    main()
