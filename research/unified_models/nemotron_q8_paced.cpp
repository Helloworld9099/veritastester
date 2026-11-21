// Local diagnostic driver for NVIDIA NeMo-Speech.cpp v0.3.0's installed ASR C ABI.
// The orchestrator must verify publisher and media pins before invoking this
// process. Stdout contains PRIVATE text; publish only redacted aggregates.
#include <nemo_speech/asr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using NS = std::chrono::nanoseconds;
constexpr int kRate = 16000;
constexpr int64_t kNsPerSample = 62500;
constexpr size_t kMaxSamples = 30 * kRate;
constexpr size_t kMaxClips = 12;
constexpr size_t kMaxResultsPerStep = 16;
constexpr size_t kMaxText = 16384;

void require(bool condition, const char* code) {
    if (!condition) throw std::runtime_error(code);
}

void check(nemo_speech_asr_status status, const char* code) {
    // Don't expose a library error's input text, model name or filesystem path.
    if (status != NEMO_SPEECH_ASR_OK)
        throw std::runtime_error(std::string(code) + "_status_" + std::to_string(status));
}

int64_t since(Clock::time_point epoch, Clock::time_point t = Clock::now()) {
    return std::chrono::duration_cast<NS>(t - epoch).count();
}

int parse_int(const char* value, int low, int high) {
    std::string s(value);
    require(!s.empty() && s.size() <= 9 &&
            std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "invalid_bounded_integer");
    const int n = std::stoi(s);
    require(n >= low && n <= high, "integer_outside_bound");
    return n;
}

std::vector<float> read_pcm(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(file), "cannot_open_local_pcm");
    const auto size = file.tellg();
    require(size > 0 && size <= std::streamoff(2 * kMaxSamples) && size % 2 == 0,
            "invalid_s16le_pcm_size");
    std::vector<unsigned char> raw(static_cast<size_t>(size));
    file.seekg(0);
    require(static_cast<bool>(file.read(reinterpret_cast<char*>(raw.data()),
                                       static_cast<std::streamsize>(raw.size()))),
            "cannot_read_pcm");
    require(file.peek() == std::char_traits<char>::eof(), "pcm_size_changed");
    std::vector<float> audio(raw.size() / 2);
    for (size_t i = 0; i < audio.size(); ++i) {
        const int code = static_cast<int>(raw[i * 2]) | (static_cast<int>(raw[i * 2 + 1]) << 8);
        const int signed_code = code < 32768 ? code : code - 65536;
        audio[i] = static_cast<float>(signed_code) / 32768.0f;
    }
    return audio;
}

void json_string(const std::string& text) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout.put('"');
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            std::cout.put('\\');
            std::cout.put(static_cast<char>(c));
        } else if (c < 32) {
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else {
            std::cout.put(static_cast<char>(c));
        }
    }
    std::cout.put('"');
}

using Engine = std::unique_ptr<nemo_speech_asr_recognizer, decltype(&nemo_speech_asr_destroy)>;
using Stream = std::unique_ptr<nemo_speech_asr_stream, decltype(&nemo_speech_asr_stream_close)>;
using Result = std::unique_ptr<nemo_speech_asr_result, decltype(&nemo_speech_asr_result_destroy)>;

struct Event {
    int64_t ready;
    bool final;
    int64_t processed_samples;
    std::string text;
};

std::vector<Event> drain(nemo_speech_asr_stream* stream, Clock::time_point epoch, int64_t deadline) {
    std::vector<Event> events;
    while (true) {
        require(since(epoch) < deadline, "clip_deadline_before_drain");
        nemo_speech_asr_result* raw = nullptr;
        const auto status = nemo_speech_asr_stream_next(stream, &raw);
        // This time reflects actual native result availability, not a model audio offset.
        const auto ready = since(epoch);
        Result result(raw, nemo_speech_asr_result_destroy);
        check(status, "native_next");
        require(ready < deadline, "clip_deadline_after_drain");
        if (!result) break;
        require(events.size() < kMaxResultsPerStep, "too_many_results_per_release");
        const double audio_s = nemo_speech_asr_result_audio_processed(result.get());
        require(std::isfinite(audio_s) && audio_s >= 0 && audio_s <= 31.0,
                "invalid_vendor_audio_position");
        const char* transcript = nemo_speech_asr_result_transcript(result.get(), 0);
        std::string text = transcript == nullptr ? "" : transcript;
        require(text.size() <= kMaxText, "invalid_vendor_caption_length");
        events.push_back({ready, nemo_speech_asr_result_is_final(result.get()),
                          static_cast<int64_t>(std::llround(audio_s * kRate)), std::move(text)});
    }
    return events;
}

void results(const std::vector<Event>& events, size_t clip, size_t seq) {
    for (const auto& event : events) {
        std::cout << "{\"kind\":\"result\",\"clip\":" << clip << ",\"seq\":" << seq
                  << ",\"ready_ns\":" << event.ready << ",\"final\":"
                  << (event.final ? "true" : "false") << ",\"processed_samples\":"
                  << event.processed_samples << ",\"text\":";
        json_string(event.text);
        std::cout << "}\n";
    }
}

void run_clip(nemo_speech_asr_recognizer* engine, size_t clip, const std::string& locale,
              const std::vector<float>& audio, size_t chunk, int64_t deadline_ns) {
    const int64_t capture_end = static_cast<int64_t>(audio.size()) * kNsPerSample;
    require(capture_end < deadline_ns, "deadline_does_not_allow_final");
    auto opts = nemo_speech_asr_recognition_options_default();
    opts.language_code = locale.c_str();
    opts.interim_results = true;
    opts.enable_word_time_offsets = false;
    nemo_speech_asr_stream* raw_stream = nullptr;
    const auto open_status = nemo_speech_asr_streaming_recognize(engine, &opts, &raw_stream);
    Stream stream(raw_stream, nemo_speech_asr_stream_close);
    check(open_status, "native_open_stream");
    require(static_cast<bool>(stream), "null_vendor_stream");

    std::cout << "{\"kind\":\"clip_start\",\"clip\":" << clip << ",\"locale\":";
    json_string(locale);
    std::cout << ",\"samples\":" << audio.size() << "}\n" << std::flush;
    const auto epoch = Clock::now();
    size_t seq = 0, released = 0;
    for (; released < audio.size(); ++seq) {
        const size_t count = std::min(chunk, audio.size() - released);
        const size_t end = released + count;
        const int64_t due = static_cast<int64_t>(end) * kNsPerSample;
        // Absolute schedule: elapsed inference time never shifts future releases.
        std::this_thread::sleep_until(epoch + NS(due));
        const auto dispatch = since(epoch);
        require(dispatch >= due && dispatch < deadline_ns, "invalid_or_exhausted_dispatch_time");
        check(nemo_speech_asr_stream_push_f32(stream.get(), audio.data() + released, count, kRate),
              "native_push");
        const auto push_return = since(epoch);
        auto events = drain(stream.get(), epoch, deadline_ns);
        const auto done = since(epoch);
        require(std::none_of(events.begin(), events.end(), [](const Event& e) { return e.final; }),
                "unexpected_final_before_explicit_eof");
        std::cout << "{\"kind\":\"step\",\"clip\":" << clip << ",\"seq\":" << seq
                  << ",\"start_sample\":" << released << ",\"end_sample\":" << end
                  << ",\"due_ns\":" << due << ",\"dispatch_ns\":" << dispatch
                  << ",\"push_return_ns\":" << push_return << ",\"done_ns\":" << done << "}\n";
        results(events, clip, seq);
        released = end;
    }

    const auto finish_start = since(epoch);
    require(finish_start < deadline_ns, "clip_deadline_before_finish");
    // v0.3.0 does EOF compute inside finish(), then exposes its result via next().
    check(nemo_speech_asr_stream_finish(stream.get()), "native_finish");
    const auto final_events = drain(stream.get(), epoch, deadline_ns);
    const auto finish_done = since(epoch);
    require(final_events.size() == 1 && final_events.front().final, "missing_unique_eof_final");
    results(final_events, clip, seq);
    std::cout << "{\"kind\":\"clip_done\",\"clip\":" << clip << ",\"released_samples\":" << released
              << ",\"steps\":" << seq << ",\"capture_end_ns\":" << capture_end
              << ",\"finish_start_ns\":" << finish_start << ",\"finish_done_ns\":" << finish_done
              << ",\"final_count\":" << final_events.size() << "}\n" << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 7 && (argc - 5) % 2 == 0 && static_cast<size_t>((argc - 5) / 2) <= kMaxClips,
                "usage_model_chunk_samples_lookahead_max_clip_ms_locale_pcm_pairs");
        const int chunk = parse_int(argv[2], 160, 16000);
        const int lookahead = parse_int(argv[3], 0, 13);
        require(lookahead == 0 || lookahead == 1 || lookahead == 3 || lookahead == 6 || lookahead == 13,
                "unsupported_lookahead_setting");
        const int64_t deadline_ns = 1'000'000LL * parse_int(argv[4], 1000, 120000);
        for (int i = 5; i < argc; i += 2)
            require(std::string(argv[i]) == "es-US" || std::string(argv[i]) == "fr-FR",
                    "unsupported_locale");
        cpu_set_t mask;
        CPU_ZERO(&mask);
        require(sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) == 2,
                "exactly_two_cpu_affinity_required");
        const std::string api = nemo_speech_asr_version();
        require(api == "nemo-speech-asr 0.3.0", "unexpected_native_api_version");
        const nemo_speech_asr_backend_config backend{sizeof(nemo_speech_asr_backend_config), -1};
        const nemo_speech_asr_model_config model{sizeof(nemo_speech_asr_model_config), argv[1], nullptr};
        const nemo_speech_asr_streaming_config streaming{
            sizeof(nemo_speech_asr_streaming_config), 0.16f, 1.92f, 1.92f, lookahead};
        const nemo_speech_asr_endpointing_config endpointing{
            sizeof(nemo_speech_asr_endpointing_config), false, false, 800};
        nemo_speech_asr_recognizer_config cfg{};
        cfg.size = sizeof(cfg);
        cfg.backend = &backend;
        cfg.model = &model;
        cfg.streaming = &streaming;
        cfg.endpointing = &endpointing;
        const auto load_start = Clock::now();
        nemo_speech_asr_recognizer* raw_engine = nullptr;
        const auto status = nemo_speech_asr_create(&cfg, &raw_engine);
        Engine engine(raw_engine, nemo_speech_asr_destroy);
        const auto load_ns = since(load_start);
        check(status, "native_model_load");
        require(static_cast<bool>(engine), "null_vendor_recognizer");
        std::cout << "{\"kind\":\"model\",\"load_ns\":" << load_ns << ",\"api_version\":";
        json_string(api);
        std::cout << ",\"cpu_affinity_count\":" << CPU_COUNT(&mask) << "}\n" << std::flush;
        for (int i = 5; i < argc; i += 2)
            run_clip(engine.get(), static_cast<size_t>((i - 5) / 2), argv[i],
                     read_pcm(argv[i + 1]), static_cast<size_t>(chunk), deadline_ns);
        require(static_cast<bool>(std::cout), "output_stream_failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "paced_native: " << error.what() << "\n";
        return 2;
    }
}
