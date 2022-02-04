from typing import Optional

import ray
from classutilities import classproperty
from ray.actor import ActorHandle


@ray.remote
class RayObjectStoreManager:

    def __init__(self):
        self.store_manager = {}

    def register_object(self, object_key: str, object: object):
        object_reference = ray.put(object)
        self.store_manager[object_key] = object_reference

    def obtain_object_reference(self, object_key: str):
        return self.store_manager[object_key]

    def delete_object_in_store(self, object_key: str):
        del self.store_manager[object_key]


DEFAULT_STORE_MANAGER_NAME = "ray_store_manager_actor"


class RayObjectStoreAccessor:
    _global_object_store_handle: Optional[ActorHandle] = None

    @classmethod
    def _register_new_store_manager(cls) -> ActorHandle:
        cls._global_object_store_handle = RayObjectStoreManager.options(
            name=DEFAULT_STORE_MANAGER_NAME).remote()
        return cls._global_object_store_handle

    # noinspection PyNestedDecorators
    @classproperty
    @classmethod
    def global_object_store_handle(cls):
        if cls._global_object_store_handle is not None:
            return cls._global_object_store_handle
        else:
            try:
                cls._global_object_store_handle = ray.get_actor(name=DEFAULT_STORE_MANAGER_NAME)
            except ValueError as e:
                cls._global_object_store_handle = cls._register_new_store_manager()
            return cls._global_object_store_handle

    @classmethod
    def save_object_in_manager(cls, key: str, obj: object) -> None:
        cls.global_object_store_handle.register_object.remote(key, obj)

    @classmethod
    def get_object_in_manager(cls, key: str) -> object:
        object_reference_remote = cls.global_object_store_handle.obtain_object_reference.remote(key)
        object_reference = ray.get(object_reference_remote)
        object_ = ray.get(object_reference)
        return object_

    @classmethod
    def delete_object_in_manager(cls, key: str) -> None:
        cls.global_object_store_handle.delete_object_in_store.remote(key)
