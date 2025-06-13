import time
from importlib.resources import read_binary
from pathlib import Path

from mcap.reader import make_reader

# import shutil
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import Writer

from airbot_type.FloatArray import FloatArray


class MCAPConverter:
    def __init__(self, source_path: Path):
        self.source_path = source_path
        if not self.source_path.is_file():
            raise ValueError(f"Source path {source_path} is not a valid file.")
        self.f = self.source_path.open("rb")
        self.reader = make_reader(self.f)
        self.summary = self.reader.get_summary()
        self.size = min(self.summary.statistics.channel_message_counts.values())
        self.low_dim_topics = [
            channel.topic
            for channel in self.summary.channels.values()
            if self.summary.statistics.channel_message_counts[channel.id] == self.size
        ]

    def __del__(self):
        if hasattr(self, "f") and not self.f.closed:
            self.f.close()

    def add_time_index(self, dest_path: Path = None) -> bool:
        print(f"Processing file: {self.source_path}")
        time_index: list[tuple[int, int]] = []
        cnt = {topic: 0 for topic in self.low_dim_topics}
        index = 0
        min_time = float("inf")
        max_time = 0
        for schema_obj, channel_obj, message_obj in self.reader.iter_messages(
            self.low_dim_topics
        ):
            cnt[channel_obj.topic] += 1
            if cnt[channel_obj.topic] - index > 1:
                time_index.append((min_time, max_time))
                min_time = float("inf")
                max_time = 0
                index += 1
            current_time = message_obj.log_time
            # current_time = message_obj.publish_time
            min_time = min(min_time, current_time)
            max_time = max(max_time, current_time)
        time_index.append((min_time, max_time))
        # print(time_index)
        if dest_path:
            # # cp the original file to the destination
            # if str(dest_path).endswith(".mcap"):
            #     dest_path = dest_path.with_suffix(".mcap")
            # dest_path.parent.mkdir(parents=True, exist_ok=True)
            # shutil.copy(self.source_path, dest_path)

            with dest_path.open("wb") as f:
                writer = Writer(f)
                writer.start()
                float_array_schema_id = writer.register_schema(
                    name="airbot_fbs.FloatArray",
                    encoding=SchemaEncoding.Flatbuffer,
                    data=read_binary(
                        "bfbs",
                        "FloatArray.bfbs",
                    ),
                )
                for topic in self.summary.channels.values():
                    writer.register_channel(
                        schema_id=float_array_schema_id,
                        topic=topic.topic,
                        message_encoding=MessageEncoding.Flatbuffer,
                    )
                for metadata in self.reader.iter_metadata():
                    writer.add_metadata(name=metadata.name, data=metadata.metadata)
                for schema_obj, channel_obj, message_obj in self.reader.iter_messages():
                    writer.add_message(
                        channel_id=channel_obj.id,
                        log_time=message_obj.log_time,
                        # log_time=message_obj.publish_time,
                        data=message_obj.data,
                        publish_time=message_obj.publish_time,
                    )
                for attachment in self.reader.iter_attachments():
                    writer.add_attachment(
                        create_time=attachment.create_time,
                        log_time=attachment.log_time,
                        data=attachment.data,
                        name=attachment.name,
                        media_type=attachment.media_type,
                    )

                # write the time index to the destination file as attachment
                writer.add_attachment(
                    time.time_ns(),
                    time.time_ns(),
                    name="time_index",
                    data=bytes(str(time_index), "utf-8"),
                    media_type="application/octet-stream",
                )
                writer.finish()
                f.close()
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MCAP files to FloatArray format."
    )
    parser.add_argument(
        "source_directory", type=str, help="Directory containing the source MCAP files."
    )
    parser.add_argument(
        "dest_directory",
        type=str,
        help="Directory to save the converted FloatArray files.",
    )
    args = parser.parse_args()
    for source_path in Path(args.source_directory).glob("*.mcap"):
        converter = MCAPConverter(source_path)
        dest_path = Path(args.dest_directory) / (source_path.stem + ".mcap")
        if converter.add_time_index(dest_path):
            print(f"Successfully converted {source_path} to {dest_path}")
        else:
            print(f"Failed to convert {source_path}")
        # converter.add_time_index()
