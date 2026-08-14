CREATE TABLE "LiteLLM_ShadowEvalJobKey" (
    "id" TEXT NOT NULL,
    "job_id" TEXT NOT NULL,
    "api_key_id" TEXT NOT NULL,
    "max_turns" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "stopped_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_ShadowEvalJobKey_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "LiteLLM_ShadowEvalJobKey_job_id_idx" ON "LiteLLM_ShadowEvalJobKey"("job_id");

CREATE INDEX "LiteLLM_ShadowEvalJobKey_api_key_id_idx" ON "LiteLLM_ShadowEvalJobKey"("api_key_id");

ALTER TABLE "LiteLLM_ShadowEvalJobKey" ADD CONSTRAINT "LiteLLM_ShadowEvalJobKey_job_id_fkey"
    FOREIGN KEY ("job_id") REFERENCES "LiteLLM_ShadowEvalJob"("id") ON DELETE CASCADE ON UPDATE CASCADE;


INSERT INTO "LiteLLM_ShadowEvalJobKey" ("id", "job_id", "api_key_id", "max_turns", "created_at", "stopped_at")
SELECT 'jobkey_' || "id", "id", "api_key_id", "max_turns", "created_at", "stopped_at"
FROM "LiteLLM_ShadowEvalJob";


ALTER TABLE "LiteLLM_ShadowEvalAttempt" ADD COLUMN "api_key_id" TEXT;

UPDATE "LiteLLM_ShadowEvalAttempt" a SET "api_key_id" = j."api_key_id"
FROM "LiteLLM_ShadowEvalJob" j WHERE a."job_id" = j."id";

-- Reachable only by an attempt whose job row is already gone, which no read path can return
DELETE FROM "LiteLLM_ShadowEvalAttempt" WHERE "api_key_id" IS NULL;

ALTER TABLE "LiteLLM_ShadowEvalAttempt" ALTER COLUMN "api_key_id" SET NOT NULL;

DROP INDEX "LiteLLM_ShadowEvalAttempt_job_id_idx";

CREATE INDEX "LiteLLM_ShadowEvalAttempt_job_id_api_key_id_idx" ON "LiteLLM_ShadowEvalAttempt"("job_id", "api_key_id");


DROP INDEX "LiteLLM_ShadowEvalJob_one_active_per_key";

CREATE UNIQUE INDEX "LiteLLM_ShadowEvalJobKey_one_active_per_key"
    ON "LiteLLM_ShadowEvalJobKey"("api_key_id") WHERE "stopped_at" IS NULL;

DROP INDEX "LiteLLM_ShadowEvalJob_api_key_id_idx";

ALTER TABLE "LiteLLM_ShadowEvalJob" DROP COLUMN "api_key_id",
    DROP COLUMN "max_turns",
    DROP COLUMN "stopped_at";
