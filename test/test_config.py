import yaml

from fedmind.config import get_config


def test_config():
    config = get_config("config.yaml")

    print("Get config from config.yaml:\n" + "-" * 30)
    print(yaml.dump(config.to_dict()))
    print("=" * 30)


if __name__ == "__main__":
    test_config()
