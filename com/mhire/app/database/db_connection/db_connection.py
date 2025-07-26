import logging
from motor.motor_asyncio import AsyncIOMotorClient
from com.mhire.app.config.config import Config

logger = logging.getLogger(__name__)

class DBConnection:
    def __init__(self):
        self.config = Config()
        try:
            self.client = AsyncIOMotorClient(self.config.mongodb_uri)
            self.db = self.client[self.config.mongodb_db]
            self.collection = self.db[self.config.mongodb_collection]
            logger.info("MongoDB client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            raise

    async def ping(self):
        """Check if MongoDB server is reachable."""
        try:
            result = await self.client.admin.command('ping')
            logger.info("MongoDB ping successful.")
            return result
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return None

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB client closed.")