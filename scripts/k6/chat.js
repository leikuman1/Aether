// Offline chat load: one unique thread_id per iteration so requests do not 409.
// The parent process injects AETHER_BASE_URL, AETHER_VUS, AETHER_ITERATIONS,
// AETHER_MAX_DURATION, AETHER_TIMEOUT and AETHER_QUESTION.

import http from "k6/http";
import { check } from "k6";

const vus = Number(__ENV.AETHER_VUS || 20);
const iterations = Number(__ENV.AETHER_ITERATIONS || 40);
const maxDuration = __ENV.AETHER_MAX_DURATION || "180s";
const question =
  __ENV.AETHER_QUESTION ||
  "BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？";

export const options = {
  summaryTrendStats: ["avg", "min", "med", "max", "p(50)", "p(90)", "p(95)", "p(99)"],
  scenarios: {
    chat: {
      executor: "shared-iterations",
      vus: vus,
      iterations: iterations,
      maxDuration: maxDuration,
    },
  },
};

export default function () {
  const base = __ENV.AETHER_BASE_URL;
  const payload = JSON.stringify({
    question: question,
    thread_id: `load-${__VU}-${__ITER}`,
  });
  const response = http.post(`${base}/v1/chat`, payload, {
    headers: { "Content-Type": "application/json" },
    timeout: __ENV.AETHER_TIMEOUT || "60s",
  });
  check(response, {
    "status is 200": (item) => item.status === 200,
  });
}

export function handleSummary(data) {
  const body = JSON.stringify(data);
  const outputs = { stdout: body + "\n" };
  if (__ENV.AETHER_K6_SUMMARY) {
    outputs[__ENV.AETHER_K6_SUMMARY] = body;
  }
  return outputs;
}
