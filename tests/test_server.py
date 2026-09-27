import concurrent.futures
import importlib.util
from pathlib import Path
import pickle
from queue import Queue
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_simd import policy_server


class Abort(Exception):
    pass


class Context:
    def peer(self):
        return "test"

    def abort(self, code, message):
        raise Abort(code, message)


class Adapter:
    camera_keys = []

    def __init__(self, engine, features):
        pass

    def __call__(self, obs):
        return obs


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "requires the serve environment")
class Server(unittest.TestCase):
    def setUp(self):
        from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
        from lerobot.transport import services_pb2
        self.pb = services_pb2
        self.observation = TimedObservation(1.0, 1, {"state": np.array([0], np.float32)}, True)
        self.engine = SimpleNamespace(chunk=2, predict=lambda *_args, **_kw: np.ones((2, 1), np.float32))
        spec = SimpleNamespace(policy_type="act", adapter_cls=Adapter)
        cls, _ = policy_server.build_servicer_class(spec)
        cfg = SimpleNamespace(rtc_horizon=0, checkpoint=None, seed=0, environment_dt=0.1,
                              obs_queue_timeout=0.1)
        self.server = cls(self.engine, cfg)
        self.context = Context()
        self.server.Ready(None, self.context)
        setup = RemotePolicyConfig("act", "test", {}, 2)
        self.server.SendPolicyInstructions(SimpleNamespace(data=pickle.dumps(setup)), self.context)

    def test_actions_and_retry_after_failure(self):
        def fail(*_args):
            raise ValueError("failed inference")

        predict = self.engine.predict
        self.engine.predict = fail
        self.server._enqueue(self.observation, None)
        with self.assertLogs(policy_server.logger, level="ERROR"), self.assertRaises(Abort):
            self.server.GetActions(None, self.context)
        self.assertNotIn(1, self.server._predicted_timesteps)
        self.assertIsNone(self.server.last_processed_obs)
        self.engine.predict = predict
        self.server._enqueue(self.observation, None)
        result = pickle.loads(self.server.GetActions(None, self.context).data)
        self.assertEqual([a.timestep for a in result], [1, 2])
        np.testing.assert_array_equal(result[0].action.numpy(), [1])

    def test_reconnect_discards_waiting_request(self):
        popped, release = threading.Event(), threading.Event()

        class PausedQueue(Queue):
            def get(self, *args, **kwargs):
                value = super().get(*args, **kwargs)
                popped.set()
                if not release.wait(5):
                    raise TimeoutError("test worker not released")
                return value

        self.server.observation_queue = PausedQueue()
        self.server.observation_queue.put((self.observation, None, None))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.server.GetActions, None, self.context)
            try:
                self.assertTrue(popped.wait(5))
                self.server.Ready(None, self.context)
            finally:
                release.set()
            self.assertIsInstance(future.result(timeout=5), self.pb.Empty)
        self.assertEqual(self.server.n_queries, 0)
        self.assertIsNone(self.server.adapter)

    def test_observations_require_instructions(self):
        self.server.Ready(None, self.context)
        from lerobot.transport.utils import send_bytes_in_chunks
        stream = send_bytes_in_chunks(pickle.dumps(self.observation), self.pb.Observation)
        with self.assertRaises(Abort):
            self.server.SendObservations(stream, self.context)


if __name__ == "__main__":
    unittest.main()
